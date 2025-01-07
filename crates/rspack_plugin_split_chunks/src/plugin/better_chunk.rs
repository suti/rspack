use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::mem;
use std::ops::Not;
use std::sync::{Arc, LazyLock};

use rayon::iter::{ParallelBridge, ParallelIterator};
use regex::Regex;
use rspack_collections::{
  Identifiable, Identifier, IdentifierDashSet, IdentifierMap, IdentifierSet, UkeyMap, UkeySet,
};
use rspack_core::incremental::Mutation;
use rspack_core::{
  compare_chunks_with_graph, compare_modules_by_identifier, merge_runtime, Chunk,
  ChunkGroupOrderKey, ChunkGroupUkey, ChunkUkey, Compilation, CompilerOptions, DependenciesBlock,
  Logger, Module, ModuleIdentifier, ModuleType, Plugin, SourceType,
};
use rspack_hash::{RspackHash, RspackHashDigest};
use rspack_util::identifier::make_paths_relative;
use rustc_hash::FxHashMap;

use crate::{SplitChunkSizes, SplitChunksPlugin};

static REPLACE_MODULE_IDENTIFIER_REG: LazyLock<Regex> =
  LazyLock::new(|| Regex::new(r"^.*!|\?[^?!]*$").expect("regexp init failed"));
static REPLACE_RELATIVE_PREFIX_REG: LazyLock<Regex> =
  LazyLock::new(|| Regex::new(r"^(\.\.?\/)+").expect("regexp init failed"));
static REPLACE_ILLEGEL_LETTER_REG: LazyLock<Regex> =
  LazyLock::new(|| Regex::new(r"(^[.-]|[^a-zA-Z0-9_-])+").expect("regexp init failed"));

fn request_to_id(req: &str) -> String {
  let mut res = REPLACE_RELATIVE_PREFIX_REG.replace_all(req, "").to_string();
  res = REPLACE_ILLEGEL_LETTER_REG
    .replace_all(&res, "_")
    .to_string();
  res
}

fn hash_filename(filename: &str, options: &CompilerOptions) -> String {
  let mut filename_hash = RspackHash::from(&options.output);
  filename.hash(&mut filename_hash);
  let hash_digest: RspackHashDigest = filename_hash.digest(&options.output.hash_digest);
  hash_digest.rendered(8).to_string()
}

fn get_size(module: &Box<dyn Module>, compilation: &Compilation) -> SplitChunkSizes {
  SplitChunkSizes(
    module
      .source_types()
      .iter()
      .map(|ty| (*ty, module.size(Some(ty), Some(compilation))))
      .collect(),
  )
}

#[derive(Debug, Clone)]
pub struct BetterChunkStage {
  pub shared: Vec<String>,
  pub entries: Vec<(String, Option<(ChunkGroupOrderKey, usize)>)>,
  pub modules: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BetterChunkOptions {
  pub strict: bool,
  pub log: bool,
  pub stages: Vec<BetterChunkStage>,
  pub keep_magic_chunks: Vec<ChunkGroupOrderKey>,
  pub remove_duplicate_modules: bool,
  pub concat_small_chunks: bool,
  pub concat_unrelated_chunks: bool,
  pub split_big_chunks: bool,
  pub split_chunk_sizes: Option<(u32, u32)>,
  pub concat_chunk_sizes: Option<(u32, u32)>,
}

#[derive(Debug, Clone)]
struct ModuleItem {
  module: ModuleIdentifier,
  size: SplitChunkSizes,
  key: String,
  module_type: ModuleType,
  is_shared: bool,
  skip_duplicate_check: bool,
}

impl Hash for ModuleItem {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    self.module.hash(state);
  }
}

impl PartialEq for ModuleItem {
  fn eq(&self, other: &Self) -> bool {
    self.module == other.module
  }
}

impl Eq for ModuleItem {
  fn assert_receiver_is_total_eq(&self) {
    self.module.assert_receiver_is_total_eq();
  }
}

type ARCCache<T> = Arc<RefCell<T>>;

fn find_parent_chunks(chunk_key: ChunkUkey, compilation: &Compilation) -> Vec<ChunkUkey> {
  let chunk_graph = &compilation.chunk_graph;
  let entry_modules = chunk_graph.get_chunk_entry_modules(&chunk_key);
  let module_graph = compilation.get_module_graph();
  entry_modules
    .iter()
    .map(|&module| {
      module_graph
        .get_incoming_connections_by_origin_module(&module)
        .values()
        .flatten()
        .map(|dep| dep.original_module_identifier)
        .filter_map(|mi| mi)
        .map(|module| {
          chunk_graph
            .get_module_chunks(module)
            .iter()
            .filter(|&&chunk| chunk != chunk_key)
        })
        .flatten()
        .cloned()
        .collect::<Vec<_>>()
    })
    .flatten()
    .collect()
}

#[inline]
fn find_parent_chunk_path(
  chunk_key: ChunkUkey,
  compilation: &Compilation,
  end_points: UkeySet<ChunkUkey>,
  cache: ARCCache<FxHashMap<ChunkUkey, Vec<Vec<ChunkUkey>>>>,
) -> Vec<Vec<ChunkUkey>> {
  if end_points.contains(&chunk_key) {
    return vec![vec![chunk_key]];
  }
  let parent_chunks = find_parent_chunks(chunk_key, compilation);
  if parent_chunks.len() == 0 {
    return vec![vec![chunk_key]];
  }
  parent_chunks
    .into_iter()
    .enumerate()
    .map(|(_index, chunk)| {
      let mut end_points = end_points.clone();
      end_points.insert(chunk_key); // 阻止循环
      let cache = cache.clone();
      let mut next_paths = {
        let cache = cache.borrow();
        cache.get(&chunk).cloned()
      }
      .map_or_else(
        || {
          let paths = find_parent_chunk_path(chunk, compilation, end_points, cache.clone());
          {
            let mut cache = cache.borrow_mut();
            cache.insert(chunk, paths.clone());
          }
          paths
        },
        |v| v,
      );
      next_paths.iter_mut().for_each(|path| {
        path.push(chunk_key);
      });
      next_paths
    })
    .flatten()
    .collect::<Vec<_>>()
}

fn find_best_chunk_for_module(
  module_id: ModuleIdentifier,
  modules_parent_chunk_path_cache: ARCCache<IdentifierMap<Vec<Vec<ChunkUkey>>>>,
) -> (ChunkUkey, Vec<ChunkUkey>) {
  let cache = modules_parent_chunk_path_cache.borrow();
  let mut paths = cache.get(&module_id).cloned().expect("");
  paths.sort_by(|a, b| a.len().cmp(&b.len()));
  let best_chunk_parent_path = paths.first().expect("");
  (
    best_chunk_parent_path.first().expect("").clone(),
    paths
      .iter()
      .enumerate()
      .filter_map(|(i, paths)| (i != 0).then(|| paths.first().cloned()).flatten())
      .collect(),
  )
}

fn find_best_parent_and_orphans(
  paths: &Vec<Vec<ChunkGroupUkey>>,
) -> (UkeySet<ChunkGroupUkey>, UkeySet<ChunkGroupUkey>) {
  // 用于记录每个节点出现的次数
  let mut node_count: UkeyMap<ChunkGroupUkey, usize> = Default::default();

  // 遍历所有路径，统计每个节点的出现次数
  for path in paths.iter() {
    for &node in path {
      *node_count.entry(node).or_insert(0) += 1;
    }
  }

  // 用于存储最佳父节点到路径的映射
  let mut result: UkeySet<ChunkGroupUkey> = Default::default();
  // 用于存储没有最佳父节点的路径
  let mut orphans: UkeySet<ChunkGroupUkey> = Default::default();

  // 遍历每一条路径，找到最佳父节点（最远的公共祖先）
  for path in paths.iter() {
    let mut best_parent = None;

    // 从路径的开头向后查找最远的公共父节点, 最多找3个
    for (index, node) in path.iter().enumerate() {
      if index > 2 {
        break;
      }
      if let Some(&count) = node_count.get(node) {
        if count > 1 {
          best_parent = Some(*node);
        }
      }
    }

    // 如果找到最佳父节点，将路径记录到结果中；否则归类为孤立路径
    if let Some(parent) = best_parent {
      result.insert(parent);
    } else {
      path.last().map(|&node| orphans.insert(node));
    }
  }

  (result, orphans)
}

#[derive(Debug, Default)]
struct ChunkMutationItem {
  chunk_key: ChunkUkey,
  modules: IdentifierSet,
  duplicate_modules: IdentifierSet,
  chunk_entries: IdentifierSet,
  duplicate_modules_priority: IdentifierMap<usize>,
  removed_modules: IdentifierSet,
  appended_chunk: UkeySet<ChunkUkey>,
  appended_to: Option<ChunkUkey>,
  removed: bool,
  skipped: bool,
  priority: usize,
  create_by_split: bool,
}

impl ChunkMutationItem {
  fn is_empty(&self) -> bool {
    self.removed_modules.len() == self.modules.len() && self.appended_chunk.len() == 0
      || self.removed
  }

  fn modules_size(&self, modules: &IdentifierMap<ModuleItem>) -> SplitChunkSizes {
    self
      .modules
      .iter()
      .filter(|m| !self.removed_modules.contains(&m))
      .fold(SplitChunkSizes::empty(), |mut l, c| {
        l.add_by(&modules.get(c).expect("").size);
        l
      })
  }

  fn get_modules<'a>(&'a self, modules: &'a IdentifierMap<ModuleItem>) -> Vec<&'a ModuleItem> {
    self
      .modules
      .iter()
      .filter_map(|module_id| {
        let module = modules.get(module_id).expect("");
        self
          .removed_modules
          .contains(&module_id)
          .not()
          .then_some(module)
      })
      .collect()
  }

  fn chunk_size(
    &self,
    chunks: &UkeyMap<ChunkUkey, ChunkMutationItem>,
    modules: &IdentifierMap<ModuleItem>,
  ) -> SplitChunkSizes {
    self
      .appended_chunk
      .iter()
      .fold(self.modules_size(modules), |mut l, c| {
        l.add_by(&chunks.get(c).unwrap().modules_size(modules));
        l
      })
  }

  fn append_chunk(&mut self, chunk: ChunkUkey) {
    self.appended_chunk.insert(chunk);
  }

  fn duplicate_modules_size(&self, modules: IdentifierMap<ModuleItem>) -> SplitChunkSizes {
    self
      .duplicate_modules
      .iter()
      .fold(SplitChunkSizes::empty(), |mut l, c| {
        l.add_by(
          &self
            .modules
            .iter()
            .find(|p| **p == *c)
            .map(|p| {
              let p = modules.get(p).expect("");
              p.size.clone()
            })
            .unwrap_or_default(),
        );
        l
      })
  }

  fn raise_priority(&mut self) {
    self.priority += 1;
  }

  fn remove_duplicate_modules(&mut self, module_id: ModuleIdentifier) {
    self.duplicate_modules.remove(&module_id);
    self.removed_modules.insert(module_id);
    if self.removed_modules.len() == self.modules.len() {
      self.removed = true;
    }
  }

  fn raise_duplicate_modules_priority(&mut self, module_id: ModuleIdentifier) {
    let priority = self
      .duplicate_modules_priority
      .entry(module_id)
      .or_insert(0);
    *priority += 1;
  }
}

#[derive(Debug, Clone)]
struct ChunkGroupPath {
  first: ChunkGroupUkey,
  chunk_group_paths: Vec<ChunkGroupUkey>,
  end_with_end_point: bool,
  looped: bool,
}
impl ChunkGroupPath {
  fn dedup(&mut self) {
    if self.looped || self.end_with_end_point {
      let Some(&last) = self.chunk_group_paths.last() else {
        return;
      };
      if self
        .chunk_group_paths
        .iter()
        .filter(|&&c| c == last)
        .count()
        > 1usize
      {
        self.chunk_group_paths.pop();
      }
    }
  }
  fn create(chunk_group: ChunkGroupUkey, end_with_end_point: bool, looped: bool) -> Self {
    Self {
      first: chunk_group,
      chunk_group_paths: vec![chunk_group],
      end_with_end_point,
      looped,
    }
  }

  fn merge_into(self, chunk_group_path: ChunkGroupPath) -> Vec<ChunkGroupPath> {
    self
      .chunk_group_paths
      .iter()
      .map(|_| {
        let mut chunk_group_paths = self.chunk_group_paths.clone();
        chunk_group_paths.push(chunk_group_path.first);
        ChunkGroupPath {
          first: chunk_group_path.first,
          chunk_group_paths,
          end_with_end_point: chunk_group_path.end_with_end_point,
          looped: chunk_group_path.looped,
        }
      })
      .collect()
  }

  fn find_parent_chunk_group_path(
    chunk_group_key: ChunkGroupUkey,
    end_points: UkeySet<ChunkUkey>,
    compilation: &Compilation,
    traversed_group: &mut UkeySet<ChunkGroupUkey>,
  ) -> Vec<ChunkGroupPath> {
    let mut result = Self::find_parent_chunk_group_path_inner(
      chunk_group_key,
      end_points,
      compilation,
      traversed_group,
    );
    result.iter_mut().for_each(|p| p.dedup());
    result
  }

  fn find_parent_chunk_group_path_inner(
    chunk_group_key: ChunkGroupUkey,
    end_points: UkeySet<ChunkUkey>,
    compilation: &Compilation,
    traversed_group: &mut UkeySet<ChunkGroupUkey>,
  ) -> Vec<ChunkGroupPath> {
    let mut chunk_group_path = Self::create(chunk_group_key, false, false);
    if traversed_group.contains(&chunk_group_key) {
      chunk_group_path.looped = true;
      return vec![chunk_group_path];
    }
    traversed_group.insert(chunk_group_key);
    let chunk_group = compilation.chunk_group_by_ukey.expect_get(&chunk_group_key);
    if chunk_group.chunks.iter().any(|ck| end_points.contains(ck)) {
      chunk_group_path.end_with_end_point = true;
      vec![chunk_group_path]
    } else {
      let mut chunk_group_paths: Vec<ChunkGroupPath> = Default::default();
      for &cgk in chunk_group.parents.iter() {
        let result = ChunkGroupPath::find_parent_chunk_group_path_inner(
          cgk,
          end_points.clone(),
          compilation,
          traversed_group,
        );
        chunk_group_paths = result
          .into_iter()
          .map(|p| p.merge_into(chunk_group_path.clone()))
          .flatten()
          .collect();
      }
      chunk_group_paths
    }
  }
}

#[inline]
fn find_parent_chunk_group_path(
  chunk_group_key: ChunkGroupUkey,
  end_points: UkeySet<ChunkUkey>,
  compilation: &Compilation,
  traversed_group: &mut UkeySet<ChunkGroupUkey>,
) -> Vec<Vec<ChunkGroupUkey>> {
  traversed_group.insert(chunk_group_key);
  let chunk_group = compilation.chunk_group_by_ukey.expect_get(&chunk_group_key);
  if chunk_group.chunks.iter().any(|ck| end_points.contains(ck)) {
    return vec![vec![chunk_group_key]];
  }
  chunk_group
    .parents
    .iter()
    .filter_map(|&cgk| {
      if traversed_group.contains(&cgk) {
        return None;
      }
      let mut result =
        find_parent_chunk_group_path(cgk, end_points.clone(), compilation, traversed_group);
      result.iter_mut().for_each(|cgk| cgk.push(chunk_group_key));
      Some(result)
    })
    .flatten()
    .collect()
}

#[derive(Debug)]
struct ChunkMutation {
  modules: IdentifierMap<ModuleItem>,
  chunks: UkeyMap<ChunkUkey, ChunkMutationItem>,
  duplicate_module_chunk: IdentifierMap<UkeySet<ChunkUkey>>,
  chunk_parent_chunk_group_paths_map: UkeyMap<ChunkUkey, Vec<Vec<ChunkGroupUkey>>>,
  module_chunk_map: IdentifierMap<ChunkUkey>,
  skipped_chunks: UkeySet<ChunkUkey>,
  skipped_modules: IdentifierSet,
  entry_chunks: UkeySet<ChunkUkey>,
}

fn contains_shared_module(compilation: &Compilation, chunk_key: &ChunkUkey) -> bool {
  let mg = compilation.get_module_graph();
  let modules = compilation.chunk_graph.get_chunk_modules(chunk_key, &mg);
  modules.iter().any(|module| {
    matches!(
      module.module_type(),
      ModuleType::ConsumeShared
        | ModuleType::ProvideShared
        | ModuleType::Fallback
        | ModuleType::Remote
    )
  })
}
impl ChunkMutation {
  fn create(
    compilation: &Compilation,
    delimiter: &str,
    skipped_chunks: UkeySet<ChunkUkey>,
  ) -> Self {
    let modules = {
      let context = compilation.options.context.as_ref();
      let module_graph = compilation.get_module_graph();
      module_graph
        .modules()
        .values()
        .map(|module| {
          // let module: &dyn Module = &*module;

          let is_external = module.as_external_module().is_some();
          let is_shared = matches!(
            module.module_type(),
            ModuleType::ConsumeShared
              | ModuleType::ProvideShared
              | ModuleType::Fallback
              | ModuleType::Remote
          );
          // let is_concatenated = module.as_concatenated_module().is_some();

          let name: String = if module.name_for_condition().is_some() {
            make_paths_relative(context, module.identifier().as_str())
          } else {
            REPLACE_MODULE_IDENTIFIER_REG
              .replace_all(&module.identifier(), "")
              .to_string()
          };
          // module.chunk_condition();
          let key = format!(
            "{}{}{}",
            name,
            delimiter,
            hash_filename(&name, &compilation.options)
          );
          (
            module.identifier(),
            ModuleItem {
              module: module.identifier(),
              size: get_size(module, compilation),
              key: request_to_id(&key),
              module_type: *module.module_type(),
              is_shared,
              skip_duplicate_check: is_external,
            },
          )
        })
        .collect::<IdentifierMap<ModuleItem>>()
    };
    let chunks = &compilation.chunk_by_ukey;
    let inner = chunks
      .iter()
      .par_bridge()
      .map(|(&chunk_key, _)| {
        // let chunk = compilation.chunk_by_ukey.expect_get(&chunk_key);
        // if chunk.name().is_some() {return None}
        let module_graph = compilation.get_module_graph();
        let chunk_graph = &compilation.chunk_graph;
        let mut items = compilation
          .chunk_graph
          .get_chunk_modules(&chunk_key, &module_graph);

        items.sort_unstable_by(|a, b| compare_modules_by_identifier(a, b));

        let modules = items.iter().map(|module| module.identifier()).collect();

        (
          chunk_key,
          ChunkMutationItem {
            chunk_key,
            modules,
            chunk_entries: chunk_graph
              .get_chunk_entry_modules(&chunk_key)
              .iter()
              .cloned()
              .collect(),
            duplicate_modules: Default::default(),
            duplicate_modules_priority: Default::default(),
            removed_modules: Default::default(),
            appended_chunk: Default::default(),
            appended_to: None,
            removed: false,
            priority: 0,
            skipped: skipped_chunks.contains(&chunk_key),
            create_by_split: false,
          },
        )
      })
      .collect::<UkeyMap<_, _>>();
    let skipped_modules = skipped_chunks
      .iter()
      .map(|chunk| {
        compilation
          .chunk_graph
          .get_chunk_modules_identifier(chunk)
          .clone()
      })
      .flatten()
      .collect();
    let entry_chunks: UkeySet<ChunkUkey> = compilation
      .entrypoints
      .values()
      .map(|entry| {
        compilation
          .chunk_group_by_ukey
          .get(entry)
          .map_or(Default::default(), |group| group.chunks.clone())
      })
      .flatten()
      .collect();
    Self {
      modules,
      chunks: inner,
      duplicate_module_chunk: Default::default(),
      chunk_parent_chunk_group_paths_map: Default::default(),
      module_chunk_map: Default::default(),
      skipped_chunks,
      skipped_modules,
      entry_chunks,
    }
  }

  fn new_chunk(&mut self, compilation: &mut Compilation) -> ChunkUkey {
    let chunk = Compilation::add_chunk(&mut compilation.chunk_by_ukey);
    if let Some(mutations) = compilation.incremental.mutations_write() {
      mutations.add(Mutation::ChunkAdd { chunk });
    }
    compilation.chunk_graph.add_chunk(chunk);
    self.chunks.insert(
      chunk,
      ChunkMutationItem {
        chunk_key: chunk,
        create_by_split: true,
        ..Default::default()
      },
    );
    chunk
  }

  fn set_chunk_modules(&mut self, chunk_key: &ChunkUkey, module_ids: IdentifierSet) {
    let Some(chunk_item) = self.chunks.get_mut(chunk_key) else {
      return;
    };
    chunk_item.modules = module_ids
  }

  fn move_module_to(
    &mut self,
    module_id: ModuleIdentifier,
    origin_chunk_key: ChunkUkey,
    chunk_key: ChunkUkey,
    compilation: &mut Compilation,
  ) {
  }

  fn find_duplicate_modules(&mut self, compilation: &Compilation) {
    // Step 1: 统计每个 ModuleIdentifier 出现的次数
    let mut module_count: IdentifierMap<usize> = Default::default();

    let duplicates: IdentifierSet = {
      let chunk_mutation_map = &self.chunks;
      for chunk in chunk_mutation_map.values() {
        for module_item in chunk
          .modules
          .iter()
          .map(|id| self.modules.get(id).expect(""))
        {
          if !module_item.skip_duplicate_check
            && !self.skipped_modules.contains(&module_item.module)
          {
            *module_count.entry(module_item.module.clone()).or_insert(0) += 1;
          }
        }
      }

      // Step 2: 收集所有在至少两个不同 ChunkD 中出现的 ModuleIdentifier
      module_count
        .into_iter()
        .filter(|(module_id, count)| {
          if *count < 2 {
            return false;
          }
          return true;
          // let module: &dyn Module = &**compilation.module_by_identifier(module_id).expect("");
          // let size = get_size(module, compilation);
          // let size = size.multiply(*count);
          // size.bigger_than(&SplitChunkSizes::with_initial_value(
          //   &[SourceType::JavaScript],
          //   1. * 1024.0,
          // )) // 1kb
        })
        .map(|(module, _)| module)
        .collect()
    };

    let chunk_mutation_map = &mut self.chunks;

    // Step 3: 将重复的 ModuleItem 插入到各自 ChunkD 的 duplicate_modules 中
    for chunk in chunk_mutation_map.values_mut() {
      for module_item in chunk
        .modules
        .iter()
        .map(|id| self.modules.get(id).expect(""))
      {
        if duplicates.contains(&module_item.module) {
          chunk.duplicate_modules.insert(module_item.module.clone());
        }
      }
    }
    self.duplicate_module_chunk = duplicates
      .into_iter()
      .map(|module_id| {
        (
          module_id,
          compilation.chunk_graph.get_module_chunks(module_id).clone(),
        )
      })
      .collect();
  }

  fn relink_module_to_chunk(
    origin_chunk_key: ChunkUkey,
    new_chunk_key: ChunkUkey,
    module_id: ModuleIdentifier,
    compilation: &mut Compilation,
  ) {
    let [origin_chunk, new_chunk] = compilation
      .chunk_by_ukey
      .get_many_mut([&origin_chunk_key, &new_chunk_key])
      .expect("[relink_module_to_chunk] chunk not found in compilation");
    let chunk_group_by_ukey = &mut compilation.chunk_group_by_ukey;

    origin_chunk
      .get_sorted_groups_iter(chunk_group_by_ukey)
      .for_each(|group| {
        let group = chunk_group_by_ukey.expect_get_mut(group);
        group.insert_chunk(new_chunk_key, origin_chunk_key);
        new_chunk.add_group(group.ukey);
      });
    origin_chunk
      .id_name_hints()
      .iter()
      .for_each(|id| new_chunk.add_id_name_hints(id.clone()));
    new_chunk.set_runtime(merge_runtime(&new_chunk.runtime(), &origin_chunk.runtime()));
    compilation
      .chunk_graph
      .disconnect_chunk_and_module(&origin_chunk_key, module_id);
  }

  fn remove_duplicate_modules(
    &mut self,
    compilation: &mut Compilation,
    stage_module_chunks: &Vec<IdentifierSet>,
  ) -> Option<()> {
    let mut move_module_to_entry_chunk: IdentifierMap<UkeySet<ChunkUkey>> = Default::default();
    let mut duplicate_modules: IdentifierSet = Default::default();
    self
      .duplicate_module_chunk
      .iter()
      .for_each(|(&module_id, chunks)| {
        let mut new_chunks = chunks.clone();
        new_chunks.retain(|chunk| !self.entry_chunks.contains(chunk));
        if new_chunks.len() < chunks.len() {
          new_chunks.iter().for_each(|&removed_chunk| {
            let chunk_d = self
              .chunks
              .get_mut(&removed_chunk)
              .expect("[remove_duplicate_modules] removed_chunk not found in chunk_mutation_map");
            chunk_d.remove_duplicate_modules(module_id);
          });
          move_module_to_entry_chunk.insert(module_id, new_chunks);
        } else {
          duplicate_modules.insert(module_id);
          chunks.iter().for_each(|&removed_chunk| {
            let chunk_d = self
              .chunks
              .get_mut(&removed_chunk)
              .expect("[remove_duplicate_modules] removed_chunk not found in chunk_mutation_map");
            chunk_d.remove_duplicate_modules(module_id);
          });
        }
      });

    // 去重，将首包重复的部分合并，后续大文件尽可能concat，最后可以都是零散文件
    // 非常小的零散文件，先按使用率 再按 按目录合并

    let mut module_chunk_map: IdentifierMap<ChunkUkey> = Default::default();
    // let logger = compilation.get_logger(String::from("remove_duplicate_modules"));

    // 筛出stage中的重复module，将同一stage的重复module合并到同一个chunk中
    stage_module_chunks.iter().for_each(|modules| {
      let mut stage_modules = modules
        .intersection(&duplicate_modules)
        .cloned()
        .collect::<IdentifierSet>();
      stage_modules.retain(|module| !module_chunk_map.contains_key(module));
      duplicate_modules.retain(|&module_id| !stage_modules.contains(&module_id));
      // logger.info(format!(
      //   "stage_modules {} {:#?}",
      //   stage_modules.len(),
      //   &stage_modules
      // ));
      if stage_modules.len() > 0 {
        let chunk = self.new_chunk(compilation);
        stage_modules.iter().for_each(|&module_id| {
          module_chunk_map.insert(module_id, chunk);
        })
      }
    });

    let mut groups: Vec<IdentifierSet> = Vec::new();
    let mut visited_modules: IdentifierSet = Default::default();

    // 计算重复module所在的chunk与其他chunk中重复module的个数，重合率高的部分放到同一个组中
    for module_a in &duplicate_modules {
      let chunks_a = self.duplicate_module_chunk.get(module_a)?;
      // 如果 module 已经被分组，跳过
      if visited_modules.contains(module_a) {
        continue;
      }
      // 初始化一个新组
      let mut group: IdentifierSet = Default::default();
      group.insert(module_a.clone());
      visited_modules.insert(module_a.clone());

      let mut candidates: Vec<(ModuleIdentifier, usize)> = Vec::new();

      for module_b in &duplicate_modules {
        let chunks_b = self.duplicate_module_chunk.get(module_b)?;
        // 跳过已处理的模块
        if visited_modules.contains(module_b) || module_a == module_b {
          continue;
        }

        // 计算两个模块的 chunk 集合的交集
        let intersection = chunks_a
          .intersection(chunks_b)
          .cloned()
          .collect::<UkeySet<ChunkUkey>>();
        let intersection_len = intersection.len();
        let half_chunks_a = chunks_a.len() / 2;
        let half_chunks_b = chunks_b.len() / 2;

        // 检查是否满足阈值条件
        if intersection_len > half_chunks_a
          || intersection_len > half_chunks_b
          || intersection_len > 10
        {
          candidates.push((module_b.clone(), intersection_len));
        }
      }

      // 按重合度从高到低排序
      candidates.sort_by(|a, b| b.1.cmp(&a.1));

      // 合并重合度最高的模块
      for (module_b, _) in candidates {
        group.insert(module_b.clone());
        visited_modules.insert(module_b.clone());
      }

      // 将完成的组加入到结果中
      groups.push(group);
    }

    // 每个组创建新chunk
    groups.iter().for_each(|group| {
      let chunk = self.new_chunk(compilation);
      group.iter().for_each(|&module_id| {
        module_chunk_map.insert(module_id, chunk);
      })
    });

    // split modules from origin chunk
    self.chunks.values().for_each(|chunk| {
      chunk.removed_modules.iter().for_each(|&module_id| {
        let Some(&new_chunk_ukey) = module_chunk_map.get(&module_id) else {
          return;
        };
        // Self::relink_module_to_chunk(chunk.chunk_key, new_chunk_ukey, module_id, compilation);
        let [new_chunk, origin] = compilation
          .chunk_by_ukey
          .get_many_mut([&new_chunk_ukey, &chunk.chunk_key])
          .expect("should have both chunks");
        origin.split(new_chunk, &mut compilation.chunk_group_by_ukey);
        *new_chunk.chunk_reason_mut() =
          Some(String::from("BetterChunk split from duplicate modules"));
        if let Some(mutations) = compilation.incremental.mutations_write() {
          mutations.add(Mutation::ChunkSplit {
            from: chunk.chunk_key,
            to: new_chunk_ukey,
          });
        }
      });
    });

    // move modules to entry chunk
    move_module_to_entry_chunk
      .iter()
      .for_each(|(&module_id, chunks)| {
        chunks.iter().for_each(|&chunk| {
          compilation
            .chunk_graph
            .disconnect_chunk_and_module(&chunk, module_id);
        })
      });

    // move modules to new chunk
    self.chunks.values().for_each(|chunk| {
      chunk.removed_modules.iter().for_each(|&module_id| {
        compilation
          .chunk_graph
          .disconnect_chunk_and_module(&chunk.chunk_key, module_id);
      });
    });

    // move new chunk to graph
    module_chunk_map.iter().for_each(|(&module_id, &chunk)| {
      compilation.chunk_graph.add_chunk(chunk);
      compilation
        .chunk_graph
        .connect_chunk_and_module(chunk, module_id);
      let mut module_ids: IdentifierSet = Default::default();
      module_ids.insert(module_id);
      self.set_chunk_modules(&chunk, module_ids);
    });

    self.module_chunk_map = module_chunk_map;

    Some(())
  }

  fn can_chunks_be_integrated(
    chunk_a_ukey: &ChunkUkey,
    chunk_b_ukey: &ChunkUkey,
    compilation: &Compilation,
  ) -> bool {
    if contains_shared_module(compilation, chunk_a_ukey)
      || contains_shared_module(compilation, chunk_b_ukey)
    {
      return false;
    }

    let chunk_by_ukey = &compilation.chunk_by_ukey;
    let chunk_group_by_ukey = &compilation.chunk_group_by_ukey;
    let chunk_graph = &compilation.chunk_graph;

    let chunk_a = chunk_by_ukey.expect_get(chunk_a_ukey);
    let chunk_b = chunk_by_ukey.expect_get(chunk_b_ukey);
    if chunk_a.prevent_integration() || chunk_b.prevent_integration() {
      return false;
    }

    let has_runtime_a = chunk_a.has_runtime(chunk_group_by_ukey);
    let has_runtime_b = chunk_b.has_runtime(chunk_group_by_ukey);

    // true, if a is always a parent of b
    let is_available_chunk = |a: &Chunk, b: &Chunk| {
      let mut queue = b.groups().clone().into_iter().collect::<Vec<_>>();
      while let Some(chunk_group_ukey) = queue.pop() {
        if a.is_in_group(&chunk_group_ukey) {
          continue;
        }
        let chunk_group = chunk_group_by_ukey.expect_get(&chunk_group_ukey);
        if chunk_group.is_initial() {
          return false;
        }
        for parent in chunk_group.parents_iterable() {
          queue.push(*parent);
        }
      }
      true
    };

    if has_runtime_a != has_runtime_b {
      if has_runtime_a {
        return is_available_chunk(chunk_a, chunk_b);
      } else if has_runtime_b {
        return is_available_chunk(chunk_b, chunk_a);
      } else {
        return false;
      }
    }

    if chunk_graph.get_number_of_entry_modules(chunk_a_ukey) > 0
      || chunk_graph.get_number_of_entry_modules(chunk_b_ukey) > 0
    {
      return false;
    }

    true
  }

  fn integrate_chunks(
    &mut self,
    best_chunk_key: ChunkUkey,
    small_chunk_key: ChunkUkey,
    compilation: &mut Compilation,
  ) -> Option<()> {
    let small_chunk_appended_chunk = {
      let small_chunk = self.chunks.get_mut(&small_chunk_key)?;
      small_chunk.removed = true;
      small_chunk.appended_to = Some(best_chunk_key);
      mem::take(&mut small_chunk.appended_chunk)
    };
    {
      let best_chunk = self.chunks.get_mut(&best_chunk_key)?;
      best_chunk.append_chunk(small_chunk_key);
      best_chunk.appended_chunk.extend(small_chunk_appended_chunk);
    }
    let mut chunk_graph = std::mem::take(&mut compilation.chunk_graph);
    let mut chunk_by_ukey = std::mem::take(&mut compilation.chunk_by_ukey);
    let mut chunk_group_by_ukey = std::mem::take(&mut compilation.chunk_group_by_ukey);
    let module_graph = compilation.get_module_graph();
    chunk_graph.integrate_chunks(
      &best_chunk_key,
      &small_chunk_key,
      &mut chunk_by_ukey,
      &mut chunk_group_by_ukey,
      &module_graph,
    );
    chunk_by_ukey.remove(&small_chunk_key);
    chunk_graph.remove_chunk(&small_chunk_key);
    if let Some(mutations) = compilation.incremental.mutations_write() {
      mutations.add(Mutation::ChunkRemove {
        chunk: small_chunk_key,
      });
      mutations.add(Mutation::ChunksIntegrate { to: best_chunk_key });
    }
    compilation.chunk_by_ukey = chunk_by_ukey;
    compilation.chunk_group_by_ukey = chunk_group_by_ukey;
    compilation.chunk_graph = chunk_graph;

    Some(())
  }

  fn pick_stage_chunk(
    &self,
    compilation: &Compilation,
    module_groups: &[IdentifierSet],
  ) -> Vec<ChunkUkey> {
    module_groups
      .iter()
      .map(|module_ids| {
        let chunks = module_ids
          .iter()
          .map(|&module_id| compilation.chunk_graph.get_module_chunks(module_id).clone())
          .flatten()
          .collect::<UkeySet<ChunkUkey>>();
        // chunks
        //   .into_iter()
        //   .filter(|&chunk| {
        //     let chunk_modules = compilation.chunk_graph.get_chunk_modules_identifier(&chunk);
        //     chunk_modules
        //       .iter()
        //       .all(|&module_id| module_ids.contains(&module_id))
        //   })
        //   .collect::<Vec<ChunkUkey>>()
        chunks
      })
      .flatten()
      .collect::<Vec<_>>()
  }

  fn optimize_stage_module_chunks(
    &mut self,
    compilation: &mut Compilation,
    stage_chunks: Vec<UkeySet<ChunkUkey>>,
  ) -> Vec<(usize, usize)> {
    let entries = compilation
      .entrypoints
      .values()
      .map(|cgk| {
        compilation
          .chunk_group_by_ukey
          .expect_get(cgk)
          .chunks
          .iter()
      })
      .flatten()
      .cloned()
      .collect::<UkeySet<ChunkUkey>>();

    stage_chunks
      .into_iter()
      .filter_map(|chunks| {
        let mut chunks = chunks.into_iter().collect::<Vec<_>>();
        chunks.retain(|c| !entries.contains(c));

        let mut target = chunks.pop();
        let origin_chunk_count = chunks.len();
        let mut removed_chunk_count = 0usize;
        loop {
          let Some(mut current_target) = target else {
            break;
          };
          let mut removed_chunk: UkeySet<ChunkUkey> = Default::default();
          // 遍历所有能合并到target的部分
          for &chunk in &chunks {
            if Self::can_chunks_be_integrated(&current_target, &chunk, compilation) {
              self.integrate_chunks(current_target, chunk, compilation);
              removed_chunk.insert(chunk);
            }
          }
          chunks.retain(|c| !removed_chunk.contains(c));
          removed_chunk_count += removed_chunk.len();
          removed_chunk.clear();
          // 遍历所有target能合并部分
          for &chunk in &chunks {
            if Self::can_chunks_be_integrated(&chunk, &current_target, compilation) {
              self.integrate_chunks(chunk, current_target, compilation);
              current_target = chunk;
              removed_chunk.insert(chunk);
            }
          }
          chunks.retain(|c| !removed_chunk.contains(c));
          removed_chunk_count += removed_chunk.len();
          target = chunks.pop();
        }
        Some((origin_chunk_count, removed_chunk_count))
      })
      .collect::<Vec<(usize, usize)>>()
  }

  fn concat_chunks(
    &mut self,
    size_limit: &(u32, u32),
    compilation: &mut Compilation,
    split_group_point: UkeySet<ChunkGroupUkey>,
    stage_chunks: Vec<UkeySet<ChunkUkey>>,
    concat_unrelated_chunks: bool,
  ) -> Option<()> {
    let chunk_ref = &self.chunks;
    let mut small_chunks = chunk_ref
      .values()
      .filter_map(|cd| {
        if !cd.is_empty()
          && cd.chunk_size(&self.chunks, &self.modules).total_size() < size_limit.0
          && !contains_shared_module(compilation, &cd.chunk_key)
        {
          Some(cd.chunk_key)
        } else {
          None
        }
      })
      .collect::<Vec<_>>();
    let mut module_chunk_vec: Vec<_> = self.module_chunk_map.values().cloned().collect();
    module_chunk_vec.sort();
    module_chunk_vec.dedup();

    small_chunks.sort_by(|a, b| {
      compare_chunks_with_graph(
        &compilation.chunk_graph,
        &compilation.get_module_graph(),
        a,
        b,
      )
    });

    let mut not_found_best_chunk: Vec<ChunkUkey> = Default::default();

    loop {
      let Some(small_chunk_key) = small_chunks.pop() else {
        break;
      };
      let is_in_stage = stage_chunks.iter().enumerate().find_map(|(index, chunks)| {
        if chunks.contains(&small_chunk_key) {
          Some(index)
        } else {
          None
        }
      });
      let small_chunk = compilation.chunk_by_ukey.get(&small_chunk_key)?;

      let can_be_integrated_withs = small_chunk
        .groups()
        .iter()
        .map(|&group| {
          if split_group_point.contains(&group) {
            return None;
          }
          compilation.chunk_group_by_ukey.get(&group).map(|group| {
            let mut groups = vec![group.chunks.clone()];
            // let parent_groups = group
            //   .parents_iterable()
            //   .map(|group| {
            //     compilation
            //       .chunk_group_by_ukey
            //       .get(&group)
            //       .map_or(vec![], |group| group.chunks.clone())
            //   })
            //   .collect::<Vec<_>>();
            // groups.extend(parent_groups);
            groups
          })
        })
        .flatten()
        .flatten()
        .flatten()
        .collect::<UkeySet<ChunkUkey>>();
      let mut can_be_integrated_withs = can_be_integrated_withs
        .into_iter()
        .filter(|chunk| {
          small_chunk_key != *chunk
            && compilation
              .chunk_by_ukey
              .get(chunk)
              .map_or(false, |chunk| chunk.name().is_none())
            && self.chunks.get(chunk).map_or(false, |chunk| {
              chunk.chunk_size(&self.chunks, &self.modules).total_size() < size_limit.1
            })
            && Self::can_chunks_be_integrated(chunk, &small_chunk_key, &compilation)
        })
        .collect::<Vec<_>>();
      // can_be_integrated_withs.sort_by(|a, b| {
      //   compare_chunks_with_graph(
      //     &compilation.chunk_graph,
      //     &compilation.get_module_graph(),
      //     a,
      //     b,
      //   )
      // });
      if can_be_integrated_withs.is_empty() {
        continue;
      }

      if let Some(is_in_stage) = is_in_stage {
        let stage_chunks = &stage_chunks[is_in_stage];
        can_be_integrated_withs = can_be_integrated_withs
          .into_iter()
          .filter(|chunk| stage_chunks.contains(&chunk))
          .collect()
      } else {
        let stage_chunks = &stage_chunks
          .iter()
          .flatten()
          .cloned()
          .collect::<UkeySet<ChunkUkey>>();
        // 不允许合并到entry Chunk
        can_be_integrated_withs.retain(|&chunk| !stage_chunks.contains(&chunk));
        // can_be_integrated_withs.clear()
      }

      let Some(best_chunk_key) = ({
        let mut target: Option<ChunkUkey> = None;
        for chunk_key in can_be_integrated_withs.iter().cloned() {
          target = Some(chunk_key);
          loop {
            let Some(t) = target else {
              break;
            };
            let target_item = self.chunks.get(&t)?;
            if target_item.removed {
              panic!("never panic!");
              target = target_item.appended_to;
            } else {
              break;
            }
          }
          if let Some(target) = target {
            if target != small_chunk_key {
              let chunk = self.chunks.get(&target)?;
              if chunk.chunk_size(&self.chunks, &self.modules).total_size() < size_limit.1
                && Self::can_chunks_be_integrated(&target, &small_chunk_key, &compilation)
              {
                break;
              }
            }
          }
          target = None;
        }
        target
      }) else {
        not_found_best_chunk.push(small_chunk_key);
        continue;
      };
      self.integrate_chunks(best_chunk_key, small_chunk_key, compilation);
      if {
        let best_chunk = self.chunks.get(&best_chunk_key)?;
        best_chunk
          .chunk_size(&self.chunks, &self.modules)
          .total_size()
          > size_limit.0
      } {
        small_chunks.retain(|&chunk| chunk != best_chunk_key);
      }
    }
    if !concat_unrelated_chunks {
      not_found_best_chunk.clear();
    }
    let mut small_chunks_in_stages: Vec<Vec<ChunkUkey>> = vec![vec![]; stage_chunks.len()];
    not_found_best_chunk.retain(|chunk| {
      !stage_chunks.iter().enumerate().any(|(index, chunks)| {
        if chunks.contains(&chunk) {
          small_chunks_in_stages[index].push(chunk.clone());
          true
        } else {
          false
        }
      })
    });
    small_chunks_in_stages
      .into_iter()
      .filter(|chunks| !chunks.is_empty())
      .for_each(|small_chunks| {
        self.concat_chunk_each(compilation, small_chunks, size_limit);
      });
    if !small_chunks.is_empty() {
      self.concat_chunk_each(compilation, not_found_best_chunk, size_limit)?;
    }
    Some(())
  }

  fn concat_chunk_each(
    &mut self,
    compilation: &mut Compilation,
    mut chunks: Vec<ChunkUkey>,
    size_limit: &(u32, u32),
  ) -> Option<()> {
    chunks.retain(|&chunk| {
      compilation
        .chunk_by_ukey
        .get(&chunk)
        .map_or(false, |chunk| {
          !(chunk.has_runtime(&compilation.chunk_group_by_ukey)
            || chunk.has_entry_module(&compilation.chunk_graph)
            || chunk.prevent_integration())
        })
    });
    chunks.sort_by(|a, b| {
      let a_chunk = compilation.chunk_by_ukey.expect_get(a);
      let b_chunk = compilation.chunk_by_ukey.expect_get(b);
      a_chunk
        .get_all_referenced_chunks(&compilation.chunk_group_by_ukey)
        .len()
        .cmp(
          &b_chunk
            .get_all_referenced_chunks(&compilation.chunk_group_by_ukey)
            .len(),
        )
    });
    let Some(mut integrated_chunk_key) = chunks.pop() else {
      return Some(());
    };
    let mut last_small_chunks: Vec<ChunkUkey> = vec![];
    loop {
      if let Some(small_chunk_key) = chunks.pop() {
        {
          let integrated_chunk = self.chunks.get(&integrated_chunk_key)?;
          let integrated_chunk_size = integrated_chunk.chunk_size(&self.chunks, &self.modules);
          if integrated_chunk_size.total_size() > size_limit.0 {
            integrated_chunk_key = small_chunk_key;
            continue;
          }
          let small_chunk = self.chunks.get(&small_chunk_key)?;
          let mut small_chunk_size = small_chunk.chunk_size(&self.chunks, &self.modules);
          small_chunk_size.add_by(&integrated_chunk_size);
          if small_chunk_size.total_size() > size_limit.0 {
            integrated_chunk_key = small_chunk_key;
            continue;
          }
        }

        if Self::can_chunks_be_integrated(&integrated_chunk_key, &small_chunk_key, &compilation) {
          self.integrate_chunks(integrated_chunk_key, small_chunk_key, compilation);
        } else {
          last_small_chunks.push(small_chunk_key);
        }
      } else {
        break;
      }
    }
    let Some(mut integrated_chunk_key) = last_small_chunks.pop() else {
      return Some(());
    };
    loop {
      if let Some(small_chunk_key) = last_small_chunks.pop() {
        {
          let integrated_chunk = self.chunks.get(&integrated_chunk_key)?;
          let integrated_chunk_size = integrated_chunk.chunk_size(&self.chunks, &self.modules);
          if integrated_chunk_size.total_size() > size_limit.0 {
            integrated_chunk_key = small_chunk_key;
            continue;
          }
          let small_chunk = self.chunks.get(&small_chunk_key)?;
          let mut small_chunk_size = small_chunk.chunk_size(&self.chunks, &self.modules);
          small_chunk_size.add_by(&integrated_chunk_size);
          if small_chunk_size.total_size() > size_limit.0 {
            integrated_chunk_key = small_chunk_key;
            continue;
          }
        }

        let Some(integrated_chunk_key) = last_small_chunks.iter().find_map(|&chunk| {
          if Self::can_chunks_be_integrated(&integrated_chunk_key, &small_chunk_key, &compilation) {
            Some(chunk)
          } else {
            None
          }
        }) else {
          continue;
        };
        self.integrate_chunks(integrated_chunk_key, small_chunk_key, compilation);
      } else {
        break;
      }
    }

    Some(())
  }

  fn remove_empty_chunks(&mut self, compilation: &mut Compilation) {
    let chunk_graph = &mut compilation.chunk_graph;
    let empty_chunks = compilation
      .chunk_by_ukey
      .iter()
      .filter(|(chunk_key, chunk)| {
        chunk_graph.get_number_of_chunk_modules(chunk_key) == 0
          && !chunk.has_runtime(&compilation.chunk_group_by_ukey)
          && chunk_graph.get_number_of_entry_modules(chunk_key) == 0
      })
      .map(|(&chunk_key, _)| chunk_key)
      .collect::<Vec<_>>();

    empty_chunks.iter().for_each(|chunk_ukey| {
      self.chunks.get_mut(chunk_ukey).map(|chunk| {
        chunk.removed = true;
      });
      if let Some(mut chunk) = compilation.chunk_by_ukey.remove(chunk_ukey) {
        chunk_graph.disconnect_chunk(&mut chunk, &mut compilation.chunk_group_by_ukey);
        if let Some(mutations) = compilation.incremental.mutations_write() {
          mutations.add(Mutation::ChunkRemove { chunk: *chunk_ukey });
        }
      }
    });
  }

  fn split_chunks(&mut self, size_limit: &(u32, u32), compilation: &mut Compilation) {
    let chunk_ref = &self.chunks;
    let mut big_chunks = chunk_ref
      .values()
      .filter(|cd| !cd.is_empty())
      .filter_map(|cd| {
        (cd.chunk_size(chunk_ref, &self.modules).total_size() > size_limit.1).then(|| cd.chunk_key)
      })
      .collect::<Vec<_>>();
    big_chunks.into_iter().for_each(|chunk_key| {
      let mut new_chunks_with_modules: Vec<Vec<ModuleIdentifier>> = vec![vec![]];
      let mut current_size = SplitChunkSizes::empty();
      compilation
        .chunk_graph
        .get_ordered_chunk_modules_identifier(&chunk_key)
        .iter()
        .for_each(|module_id| {
          self.modules.get(module_id).map(|module| {
            let current = new_chunks_with_modules.last_mut().unwrap();
            current.push(*module_id);
            current_size.add_by(&module.size);
            if current_size.total_size() > size_limit.0 {
              new_chunks_with_modules.push(vec![]);
              current_size = SplitChunkSizes::empty();
            }
          });
        });
      new_chunks_with_modules.retain(|module_ids| module_ids.len() > 0);
      if new_chunks_with_modules.len() > 0 {
        new_chunks_with_modules
          .iter()
          .skip(1)
          .for_each(|module_ids| {
            let new_chunk_key = self.new_chunk(compilation);
            let [origin, new] = compilation
              .chunk_by_ukey
              .get_many_mut([&chunk_key, &new_chunk_key])
              .expect("");
            origin.split(new, &mut compilation.chunk_group_by_ukey);
            if let Some(mutations) = compilation.incremental.mutations_write() {
              mutations.add(Mutation::ChunkSplit {
                from: chunk_key,
                to: new_chunk_key,
              });
            }
            module_ids.iter().for_each(|module_id| {
              compilation
                .chunk_graph
                .connect_chunk_and_module(new_chunk_key, *module_id);
              compilation
                .chunk_graph
                .disconnect_chunk_and_module(&chunk_key, *module_id);
              self.chunks.get_mut(&chunk_key).map(|chunk| {
                chunk.modules.remove(module_id);
              });
              self.chunks.get_mut(&new_chunk_key).map(|chunk| {
                chunk.modules.insert(*module_id);
              });
            });
          })
      }
    });
  }
}

fn pick_shared_module_chunk(
  compilation: &Compilation,
) -> HashMap<String, (ModuleIdentifier, ChunkUkey, IdentifierSet)> {
  let mg = compilation.get_module_graph();
  let module_map = mg
    .modules()
    .iter()
    .filter(|(_, module)| {
      matches!(
        module.source_types(),
        &[SourceType::JavaScript, SourceType::Expose]
      )
    })
    .map(|(_, module)| {
      module
        .get_blocks()
        .iter()
        .filter_map(|bi| mg.block_by_id(bi))
        .map(|block| block.get_dependencies().iter())
    })
    .flatten()
    .flatten()
    .filter_map(|dependency_id| {
      let dep = mg.dependency_by_id(dependency_id)?;
      let request = dep.as_module_dependency()?.request().to_string();
      let module_id = mg.module_identifier_by_dependency_id(dep.id())?.clone();
      let mc = compilation.chunk_graph.get_module_chunks(module_id);
      assert_eq!(mc.len(), 1);
      let chunk = mc.iter().next()?;
      let module = mg.module_by_identifier(&module_id)?;

      let children_chunks = {
        let chunk = compilation.chunk_by_ukey.get(&chunk)?;
        chunk
          .groups()
          .iter()
          .filter_map(|cgk| {
            let cg = compilation.chunk_group_by_ukey.get(cgk)?;
            Some(cg.children().iter().filter_map(|cgk| {
              let cg = compilation.chunk_group_by_ukey.get(cgk)?;
              Some(cg.chunks.clone())
            }))
          })
          .flatten()
          .flatten()
          .collect::<UkeySet<ChunkUkey>>()
      };
      // shared expose module
      let mut deps = module
        .get_dependencies()
        .into_iter()
        .filter_map(|dep| mg.module_identifier_by_dependency_id(dep))
        .cloned()
        .collect::<IdentifierSet>();
      let chunk_deps = deps
        .iter()
        .map(|&dep| {
          compilation
            .chunk_graph
            .get_module_chunks(dep)
            .into_iter()
            .filter(|ck| children_chunks.contains(ck))
            .map(|chunk| {
              compilation
                .chunk_graph
                .get_chunk_modules_identifier(chunk)
                .iter()
            })
            .flatten()
        })
        .flatten()
        .cloned()
        .collect::<IdentifierSet>();
      deps.extend(chunk_deps);
      // let chunk_group = compilation.chunk_by_ukey.get(&chunk)?.groups();
      // assert_eq!(chunk_group.len(), 1);
      Some((request, (module_id, chunk.clone(), deps)))
    })
    .collect::<HashMap<String, (ModuleIdentifier, ChunkUkey, IdentifierSet)>>();
  module_map
}

pub(crate) fn pick_entry_modules(
  compilation: &Compilation,
  entries: &Vec<(String, Option<(ChunkGroupOrderKey, usize)>)>,
) -> IdentifierSet {
  let runtimes: UkeySet<ChunkUkey> = compilation.get_chunk_graph_entries().collect();
  let logger = compilation.get_logger(String::from("pick_entry_module_chunk"));
  let chunks = entries
    .iter()
    .filter_map(|(entry_name, find_child)| {
      let entry = compilation.entrypoints.get(entry_name)?;
      let mut groups: UkeySet<ChunkGroupUkey> = Default::default();
      groups.insert(*entry);
      if let Some((find_child, mut depth)) = find_child {
        let mut current_groups = vec![*entry];
        loop {
          if depth == 0 {
            break;
          }
          depth -= 1;
          let _current_groups = current_groups.clone();
          current_groups.clear();
          _current_groups.iter().for_each(|&chunk_group| {
            let chunk_group = compilation.chunk_group_by_ukey.expect_get(&chunk_group);
            chunk_group
              .get_children_by_orders(compilation)
              .get(find_child)
              .map(|cg| {
                current_groups.extend(cg);
                let cgv: UkeySet<ChunkGroupUkey> = cg.iter().cloned().collect();
                groups.extend(&cgv);
              });
          });
          if current_groups.is_empty() {
            break;
          }
        }
      }
      // logger.info(format!(
      //   "groups {:#?}",
      //   groups
      //     .iter()
      //     .map(|&cgu| (
      //       compilation.chunk_group_by_ukey.expect_get(&cgu).name(),
      //       compilation
      //         .chunk_group_by_ukey
      //         .expect_get(&cgu)
      //         .chunks
      //         .len(),
      //       compilation
      //         .chunk_group_by_ukey
      //         .expect_get(&cgu)
      //         .chunks
      //         .iter()
      //         .map(|chunk| compilation.chunk_by_ukey.expect_get(&chunk).name())
      //         .collect::<Vec<_>>(),
      //       {
      //         let mg = compilation.get_module_graph();
      //         compilation
      //           .chunk_group_by_ukey
      //           .expect_get(&cgu)
      //           .chunks
      //           .iter()
      //           .map(|chunk| {
      //             compilation
      //               .chunk_graph
      //               .get_chunk_modules(chunk, &mg)
      //               .iter()
      //               .filter_map(|module| module.name_for_condition().map(|name| name.to_string()))
      //               .collect::<Vec<_>>()
      //           })
      //           .flatten()
      //           .collect::<Vec<_>>()
      //           .len()
      //       },
      //     ))
      //     .collect::<Vec<_>>()
      // ));
      Some(groups)
    })
    .flatten()
    .map(|cgu| {
      compilation
        .chunk_group_by_ukey
        .expect_get(&cgu)
        .chunks
        .clone()
    })
    .flatten()
    .filter(|cgu| !runtimes.contains(cgu))
    .collect::<UkeySet<ChunkUkey>>();

  let modules = chunks
    .iter()
    .map(|chunk_key| {
      compilation
        .chunk_graph
        .get_chunk_modules_identifier(chunk_key)
    })
    .flatten()
    .cloned()
    .collect::<IdentifierSet>();
  modules
}

fn pick_chunks_from_modules(
  compilation: &Compilation,
  modules: &Vec<IdentifierSet>,
) -> Vec<UkeySet<ChunkUkey>> {
  let mut visited_chunks: UkeySet<ChunkUkey> = Default::default();
  modules
    .iter()
    .map(|modules| {
      let mut chunks = modules
        .iter()
        .map(|&module| compilation.chunk_graph.get_module_chunks(module))
        .flatten()
        .cloned()
        .collect::<UkeySet<ChunkUkey>>();
      chunks.retain(|chunk| !visited_chunks.contains(chunk));
      visited_chunks.extend(&chunks);
      chunks
    })
    .collect::<Vec<_>>()
}

pub(crate) fn pick_stage_modules(
  compilation: &Compilation,
  stages: &Vec<BetterChunkStage>,
  shared_module_chunks: &HashMap<String, (ModuleIdentifier, ChunkUkey, IdentifierSet)>,
  keep_magic_chunks: &Vec<ChunkGroupOrderKey>,
  strict: bool,
) -> Vec<IdentifierSet> {
  let mut visited_modules: IdentifierSet = Default::default();
  let mut stages: Vec<IdentifierSet> = stages
    .iter()
    .map(|stage| {
      let BetterChunkStage {
        shared,
        entries,
        modules: stage_module_names,
      } = stage;
      let mut modules: IdentifierSet = Default::default();
      if modules.len() > 0 {
        let mg = compilation.get_module_graph();
        let module_ids = mg
          .modules()
          .values()
          .filter_map(|module| {
            module
              .name_for_condition()
              .map_or(false, |module_name| {
                stage_module_names
                  .iter()
                  .any(|name| module_name.contains(name))
              })
              .then_some(module.identifier())
          })
          .collect::<IdentifierSet>();
        modules.extend(module_ids);
      }
      if shared.len() > 0 {
        if strict {
          let smc_keys = shared_module_chunks.keys().collect::<Vec<_>>();
          assert!(shared.iter().all(|shared_path| smc_keys
            .iter()
            .any(|full_path| full_path.contains(shared_path))));
        }
        shared_module_chunks
          .keys()
          .filter(|full_path| {
            shared
              .iter()
              .any(|shared_path| full_path.contains(shared_path))
          })
          .filter_map(|id| shared_module_chunks.get(id))
          .for_each(|(module, chunk, deps)| {
            modules.insert(*module);
            modules.extend(deps);
          });
      }
      if entries.len() > 0 {
        let entry_ = pick_entry_modules(compilation, entries);
        modules.extend(entry_);
      }
      modules.retain(|module| !visited_modules.contains(module));
      visited_modules.extend(&modules);
      modules
    })
    .collect();
  if keep_magic_chunks.len() > 0 {
    let runtimes: UkeySet<ChunkUkey> = compilation.get_chunk_graph_entries().collect();
    keep_magic_chunks.iter().for_each(|rt| {
      compilation
        .chunk_group_by_ukey
        .values()
        .filter(|group| {
          group
            .kind
            .get_normal_options()
            .map_or(false, |options| match rt {
              ChunkGroupOrderKey::Preload => options.preload_order.is_some(),
              ChunkGroupOrderKey::Prefetch => options.preload_order.is_some(),
            })
        })
        .for_each(|cg| {
          let mut chunks = cg.chunks.iter().cloned().collect::<UkeySet<_>>();
          chunks.retain(|chunk| !runtimes.contains(chunk));
          let mut modules = chunks
            .iter()
            .map(|chunk_key| {
              compilation
                .chunk_graph
                .get_chunk_modules_identifier(chunk_key)
            })
            .flatten()
            .cloned()
            .collect::<IdentifierSet>();
          modules.retain(|module| !visited_modules.contains(module));
          visited_modules.extend(&modules);
          stages.push(modules);
        });
    });
  }
  stages
}

impl SplitChunksPlugin {
  pub(in crate::plugin) fn better_chunks(
    &self,
    compilation: &mut Compilation,
    delimiter: &str,
    skipped_chunks: UkeySet<ChunkUkey>,
    split_group_point: &Vec<String>,
    options: &BetterChunkOptions,
  ) {
    let logger = compilation.get_logger(self.name());
    let compilation_ref = &*compilation;

    let mut chunk_mutation = ChunkMutation::create(compilation, delimiter, skipped_chunks);

    let start = logger.time("make better_chunks");

    let shared_module_chunk_map = pick_shared_module_chunk(compilation);
    let strict = options.strict;
    let log = options.log;

    let stage_modules = pick_stage_modules(
      compilation,
      &options.stages,
      &shared_module_chunk_map,
      &options.keep_magic_chunks,
      strict,
    );

    if log {
      logger.info(format!("BetterChunkOptions {:#?}", &options));
      logger.info(format!(
        "stage_modules {:#?}",
        &stage_modules.iter().map(|v| v.len()).collect::<Vec<_>>()
      ));
      logger.info(format!(
        "shared_module_chunk_map {:#?}",
        &shared_module_chunk_map
          .iter()
          .map(|(k, v)| {
            (
              k.clone(),
              v.2
                .iter()
                .filter_map(|m| {
                  compilation
                    .module_by_identifier(m)
                    .unwrap()
                    .name_for_condition()
                    .map(|name| name.to_string())
                })
                .collect::<HashSet<String>>()
                .len(),
            )
          })
          .collect::<HashMap<_, _>>()
      ));
    }

    let split_group_point = split_group_point
      .iter()
      .filter_map(|end_point| {
        let chunk_key = compilation_ref.named_chunks.get(end_point)?;
        let chunk = compilation_ref.chunk_by_ukey.get(chunk_key)?;
        let mut group_points: Vec<ChunkGroupUkey> = Default::default();
        for group_key in chunk.groups() {
          let group = compilation_ref.chunk_group_by_ukey.get(group_key)?;
          let children_by_orders = group.get_children_by_orders(&compilation_ref);
          let children = children_by_orders.get(&ChunkGroupOrderKey::Preload)?;
          group_points.extend(children);
        }
        Some(group_points)
      })
      .flatten()
      .collect::<UkeySet<ChunkGroupUkey>>();

    if options.remove_duplicate_modules {
      chunk_mutation.find_duplicate_modules(compilation);

      chunk_mutation.remove_duplicate_modules(compilation, &stage_modules);

      chunk_mutation.remove_empty_chunks(compilation);

      let stage_chunks = pick_chunks_from_modules(compilation, &stage_modules);

      let optimize_result = chunk_mutation.optimize_stage_module_chunks(compilation, stage_chunks);

      optimize_result
        .iter()
        .for_each(|(origin_size, removed_size)| {
          if log {
            logger.info(format!(
              "opt_module_group {:?} -> {:?}",
              origin_size,
              origin_size - removed_size
            ));
          }
        });
    }

    if log {
      let mg = compilation.get_module_graph();
      for (index, module_ids) in stage_modules.iter().enumerate() {
        let mut module_names = vec![];
        for module_id in module_ids {
          let module = mg.module_by_identifier(&module_id).unwrap();
          if let Some(name) = module.name_for_condition() {
            module_names.push(name);
          }
        }
        logger.info(format!("stage {} {:#?}", index, &module_names.len()));
      }
    }

    if options.concat_small_chunks {
      let stage_chunks = pick_chunks_from_modules(compilation, &stage_modules);

      chunk_mutation.concat_chunks(
        &options
          .concat_chunk_sizes
          .unwrap_or((600 * 1024, 800 * 1024)),
        compilation,
        split_group_point,
        stage_chunks,
        options.concat_unrelated_chunks,
      );

      chunk_mutation.remove_empty_chunks(compilation);
    }

    if options.split_big_chunks {
      chunk_mutation.split_chunks(
        &options
          .split_chunk_sizes
          .unwrap_or((1600 * 1024, 2048 * 1024)),
        compilation,
      );

      chunk_mutation.remove_empty_chunks(compilation);
    }

    logger.time_end(start);
  }
}
