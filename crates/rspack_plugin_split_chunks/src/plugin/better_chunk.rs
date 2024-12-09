use std::cell::RefCell;
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
  ChunkGroupOrderKey, ChunkGroupUkey, ChunkUkey, Compilation, CompilerOptions, Logger, Module,
  ModuleIdentifier, ModuleType, Plugin, SourceType,
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
    end_points: UkeySet<ChunkUkey>,
  ) {
    // let end_group = end_points
    //   .iter()
    //   .map(|chunk| {
    //     let chunk = compilation.chunk_by_ukey.get(chunk).expect("");
    //     chunk.groups().clone()
    //   })
    //   .flatten()
    //   .collect::<UkeySet<ChunkGroupUkey>>();
    // self.chunk_parent_chunk_group_paths_map = self
    //   .duplicate_module_chunk
    //   .values()
    //   .flatten()
    //   .map(|&chunk_key| {
    //     let chunk = compilation.chunk_by_ukey.expect_get(&chunk_key);
    //     let mut traversed_group = UkeySet::default();
    //     let mut parent_paths = chunk
    //       .groups()
    //       .iter()
    //       .map(|&cgk| {
    //         find_parent_chunk_group_path(cgk, end_points.clone(), compilation, &mut traversed_group)
    //       })
    //       .flatten()
    //       .collect::<Vec<_>>();
    //     parent_paths.iter_mut().for_each(|path| {
    //       if path.last().map_or(false, |last| end_group.contains(last)) {
    //         path.pop();
    //       }
    //     });
    //     parent_paths.sort_by(|a, b| a.len().cmp(&b.len()));
    //     (chunk_key, parent_paths)
    //   })
    //   .collect();
    // find the best common parent group
    // insert in parent group
    // try to integrate common module chunk in same group
    // integrate chunk with other chunk in group

    // let chunk_parent_best_chunk_group_map = self
    //   .chunk_parent_chunk_group_paths_map
    //   .iter()
    //   .map(|(&chunk_key, paths)| (chunk_key, find_best_parent_and_orphans(paths)))
    //   .collect::<UkeyMap<ChunkUkey, (UkeySet<ChunkGroupUkey>, UkeySet<ChunkGroupUkey>)>>();
    // let mut module_chunk_group_map: IdentifierMap<UkeySet<ChunkGroupUkey>> = Default::default();
    let mut move_module_to_entry_chunk: IdentifierMap<UkeySet<ChunkUkey>> = Default::default();
    let mut split_modules: IdentifierSet = Default::default();
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
          // chunks.iter().for_each(|chunk| {
          //   chunk_parent_best_chunk_group_map
          //     .get(chunk)
          //     .map(|(best_parent, orphans)| {
          //       let module_chunk_group = module_chunk_group_map
          //         .entry(module_id)
          //         .or_insert(Default::default());
          //       module_chunk_group.extend(best_parent);
          //       module_chunk_group.extend(orphans);
          //     });
          // });
          split_modules.insert(module_id);
          chunks.iter().for_each(|&removed_chunk| {
            let chunk_d = self
              .chunks
              .get_mut(&removed_chunk)
              .expect("[remove_duplicate_modules] removed_chunk not found in chunk_mutation_map");
            chunk_d.remove_duplicate_modules(module_id);
          });
        }
      });

    // let mut chunk_group_module_map: UkeyMap<ChunkGroupUkey, IdentifierSet> = Default::default();
    // module_chunk_group_map
    //   .iter()
    //   .for_each(|(module, chunk_groups)| {
    //     chunk_groups.iter().for_each(|chunk_group| {
    //       chunk_group_module_map
    //         .entry(*chunk_group)
    //         .or_insert(Default::default())
    //         .insert(*module);
    //     })
    //   });
    let mut module_chunk_map: IdentifierMap<ChunkUkey> = split_modules
      .iter()
      .map(|&module_id| (module_id, self.new_chunk(compilation)))
      .collect();

    // let mut chunk_group_module_vec: Vec<(ChunkGroupUkey, &IdentifierSet)> = chunk_group_module_map
    //   .iter()
    //   .map(|(&chunk_group, modules)| (chunk_group, modules))
    //   .collect();
    // chunk_group_module_vec.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    // chunk_group_module_vec
    //   .into_iter()
    //   .for_each(|(group, module_ids)| {
    //     let module_ids = module_ids
    //       .iter()
    //       .filter(|&module_id| module_chunk_map.get(module_id).is_none())
    //       .cloned()
    //       .collect::<IdentifierSet>();
    //     if module_ids.len() > 0 {
    //       let new_chunk_ukey = self.new_chunk(compilation);
    //       module_ids.iter().for_each(|&module_id| {
    //         module_chunk_map.insert(module_id, new_chunk_ukey);
    //       });
    //       chunk_modules_map
    //         .entry(new_chunk_ukey)
    //         .or_insert(module_ids);
    //     }
    //   });

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
    // chunk_modules_map
    //   .into_iter()
    //   .for_each(|(chunk, module_ids)| {
    //     compilation.chunk_graph.add_chunk(chunk);
    //     module_ids.iter().for_each(|&module_id| {
    //       compilation
    //         .chunk_graph
    //         .connect_chunk_and_module(chunk, module_id);
    //     });
    //     self.set_chunk_modules(&chunk, module_ids);
    //   });

    self.module_chunk_map = module_chunk_map;
  }

  fn can_chunks_be_integrated(
    chunk_a_ukey: &ChunkUkey,
    chunk_b_ukey: &ChunkUkey,
    compilation: &Compilation,
    check_loop: bool,
  ) -> Option<bool> {
    if check_loop {
      let chunk_b = compilation.chunk_by_ukey.get(chunk_b_ukey)?;
      let deps_loop = chunk_b
        .get_all_referenced_chunks(&compilation.chunk_group_by_ukey)
        .contains(chunk_a_ukey);
      if deps_loop {
        return Some(false);
      }
    }

    if contains_shared_module(compilation, chunk_a_ukey)
      || contains_shared_module(compilation, chunk_b_ukey)
    {
      return Some(false);
    }

    Some(compilation.chunk_graph.can_chunks_be_integrated(
      chunk_a_ukey,
      chunk_b_ukey,
      &compilation.chunk_by_ukey,
      &compilation.chunk_group_by_ukey,
    ))
  }

  fn integrate_chunks(
    best_chunk_key: ChunkUkey,
    small_chunk_key: ChunkUkey,
    compilation: &mut Compilation,
  ) {
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
  }

  fn concat_chunks(
    &mut self,
    size_limit: &(u32, u32),
    compilation: &mut Compilation,
    split_group_point: UkeySet<ChunkGroupUkey>,
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

    // let mut module_chunk_vec: Vec<_> = self.module_chunk_map.iter().map(|(&module_id, &chunk_key)| {
    //   (module_id, chunk_key)
    // }).collect();
    //
    // module_chunk_vec.sort_by(|a, b| {
    //   self.duplicate_module_chunk.get(&a.0).map_or(0, |mc| mc.len())
    //       .cmp(&self.duplicate_module_chunk.get(&b.0).map_or(0, |mc| mc.len()))
    // });
    //
    // small_chunks = self.module_chunk_map.values().cloned().collect::<Vec<_>>();

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
            && Self::can_chunks_be_integrated(chunk, &small_chunk_key, &compilation, false)
              .expect("")
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
              target = target_item.appended_to;
            } else {
              break;
            }
          }
          if let Some(target) = target {
            if target != small_chunk_key {
              let chunk = self.chunks.get(&target)?;
              if chunk.chunk_size(&self.chunks, &self.modules).total_size() < size_limit.1
                && Self::can_chunks_be_integrated(&target, &small_chunk_key, &compilation, false)?
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
      Self::integrate_chunks(best_chunk_key, small_chunk_key, compilation);
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

    not_found_best_chunk.clear();

    not_found_best_chunk.retain(|&chunk| {
      compilation
        .chunk_by_ukey
        .get(&chunk)
        .map_or(false, |chunk| {
          !(chunk.has_runtime(&compilation.chunk_group_by_ukey)
            || chunk.has_entry_module(&compilation.chunk_graph)
            || chunk.prevent_integration())
        })
    });
    not_found_best_chunk.sort_by(|a, b| {
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
    let Some(mut integrated_chunk_key) = not_found_best_chunk.pop() else {
      return Some(());
    };
    let mut last_small_chunks: Vec<ChunkUkey> = vec![];
    loop {
      if let Some(small_chunk_key) = not_found_best_chunk.pop() {
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

        if Self::can_chunks_be_integrated(
          &integrated_chunk_key,
          &small_chunk_key,
          &compilation,
          false,
        )? {
          let small_chunk_appended_chunk = {
            let small_chunk = self.chunks.get_mut(&small_chunk_key)?;
            small_chunk.removed = true;
            small_chunk.appended_to = Some(integrated_chunk_key);
            mem::take(&mut small_chunk.appended_chunk)
          };
          {
            let integrated_chunk = self.chunks.get_mut(&integrated_chunk_key)?;
            integrated_chunk.append_chunk(small_chunk_key);
            integrated_chunk
              .appended_chunk
              .extend(small_chunk_appended_chunk);
          }
          Self::integrate_chunks(integrated_chunk_key, small_chunk_key, compilation);
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
          if Self::can_chunks_be_integrated(
            &integrated_chunk_key,
            &small_chunk_key,
            &compilation,
            false,
          )? {
            Some(chunk)
          } else {
            None
          }
        }) else {
          continue;
        };
        let small_chunk_appended_chunk = {
          let small_chunk = self.chunks.get_mut(&small_chunk_key)?;
          small_chunk.removed = true;
          small_chunk.appended_to = Some(integrated_chunk_key);
          mem::take(&mut small_chunk.appended_chunk)
        };
        {
          let integrated_chunk = self.chunks.get_mut(&integrated_chunk_key)?;
          integrated_chunk.append_chunk(small_chunk_key);
          integrated_chunk
            .appended_chunk
            .extend(small_chunk_appended_chunk);
        }
        Self::integrate_chunks(integrated_chunk_key, small_chunk_key, compilation);
      } else {
        break;
      }
    }

    Some(())
  }

  fn remove_empty_chunks(&mut self, compilation: &mut Compilation) {
    let empty_chunks = self
      .chunks
      .values()
      .filter_map(|chunk| {
        if chunk.is_empty() || chunk.removed {
          let chunk_graph = &compilation.chunk_graph;
          let chunk_key = chunk.chunk_key;
          let Some(origin_chunk) = compilation.chunk_by_ukey.get(&chunk_key) else {
            return None;
          };
          if chunk_graph.get_number_of_chunk_modules(&chunk_key) == 0
            && !origin_chunk.has_runtime(&compilation.chunk_group_by_ukey)
            && chunk_graph.get_number_of_entry_modules(&chunk_key) == 0
          {
            Some(chunk_key)
          } else {
            None
          }
        } else {
          None
        }
      })
      .collect::<Vec<_>>();

    empty_chunks.into_iter().for_each(|chunk_key| {
      let chunk_graph = &mut compilation.chunk_graph;
      if let Some(mut chunk) = compilation.chunk_by_ukey.remove(&chunk_key) {
        chunk_graph.disconnect_chunk(&mut chunk, &mut compilation.chunk_group_by_ukey);
        if let Some(mutations) = compilation.incremental.mutations_write() {
          mutations.add(Mutation::ChunkRemove { chunk: chunk_key });
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

impl SplitChunksPlugin {
  pub(in crate::plugin) fn better_chunks(
    &self,
    compilation: &mut Compilation,
    delimiter: &str,
    skipped_chunks: UkeySet<ChunkUkey>,
    split_group_point: &Vec<String>,
  ) {
    let logger = compilation.get_logger(self.name());
    let compilation_ref = &*compilation;
    let size_limit = (800 * 1024, 1600 * 1024);

    let mut chunk_mutation = ChunkMutation::create(compilation, delimiter, skipped_chunks);

    let start = logger.time("make better_chunks");

    let entry_points = compilation.entrypoints.values().map(|entrypoint_ukey| {
      let entrypoint = compilation.chunk_group_by_ukey.expect_get(entrypoint_ukey);
      entrypoint.get_runtime_chunk(&compilation.chunk_group_by_ukey)
    });

    let end_points = entry_points.collect::<UkeySet<_>>();

    let split_group_point = split_group_point
      .iter()
      .filter_map(|end_point| {
        let chunk_key = compilation_ref.named_chunks.get(end_point)?;
        let chunk = compilation_ref.chunk_by_ukey.get(chunk_key)?;
        let mut group_points: Vec<ChunkGroupUkey> = Default::default();
        logger.info(format!(
          "split_group_point_target_group {:?}",
          &chunk.groups()
        ));
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

    logger.info(format!("split_group_point {:?}", &split_group_point));

    chunk_mutation.find_duplicate_modules(compilation_ref);

    logger.info(format!(
      "find_duplicate_modules {:#?}",
      &chunk_mutation
        .duplicate_module_chunk
        .iter()
        .map(|(&a, b)| (a, b.clone()))
        .collect::<Vec<_>>()
        .sort_by(|a, b| b.1.len().cmp(&a.1.len()))
    ));

    chunk_mutation.remove_duplicate_modules(compilation, end_points.clone());

    chunk_mutation.remove_empty_chunks(compilation);

    chunk_mutation.concat_chunks(&(400 * 1024, 500 * 1024), compilation, split_group_point);

    chunk_mutation.split_chunks(&(1200 * 1024, 2048 * 1024), compilation);

    logger.info(format!(
      "entry_points {:?}",
      &compilation
        .entrypoints
        .keys()
        .map(|k| compilation.entrypoint_by_name(k).chunks.clone())
        .collect::<Vec<_>>()
    ));

    logger.time_end(start);
  }
}
