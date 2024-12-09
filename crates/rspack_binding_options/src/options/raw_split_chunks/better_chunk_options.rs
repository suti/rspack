use std::collections::HashMap;

use napi::Either;
use napi_derive::napi;
use rspack_core::ChunkGroupOrderKey;
use rspack_plugin_split_chunks::{BetterChunkOptions, BetterChunkStage, SplitChunkSizes};

#[derive(Debug)]
#[napi(object, object_to_js = false)]
pub struct RawStage {
  pub shared: Option<Vec<String>>,
  pub entries: Option<Either<HashMap<String, Option<String>>, Vec<String>>>,
  pub modules: Option<Vec<String>>,
}

#[derive(Debug)]
#[napi(object, object_to_js = false)]
pub struct RawBetterChunkOptions {
  pub strict: Option<bool>,
  pub log: Option<bool>,
  pub keep_named_cache_group: Option<bool>,
  pub keep_named_chunk: Option<bool>,
  pub remove_duplicate_modules: Option<bool>,
  pub stages: Option<Vec<RawStage>>,
  pub keep_magic_chunks: Option<Vec<String>>,
  pub concat_small_chunks: Option<bool>,
  pub concat_unrelated_chunks: Option<bool>,
  pub skip_concat_chunks: Option<Vec<String>>,
  pub split_big_chunks: Option<bool>,
  pub split_chunk_sizes: Option<Vec<u32>>,
  pub concat_chunk_sizes: Option<Vec<u32>>,
}

fn vec_to_tuple<T>(v: Vec<T>) -> (T, T)
where
  T: Copy,
{
  (v[0], v[1])
}

impl From<RawBetterChunkOptions> for BetterChunkOptions {
  fn from(options: RawBetterChunkOptions) -> Self {
    Self {
      strict: options.strict.unwrap_or(false),
      log: options.log.unwrap_or(false),
      keep_named_cache_group: options.keep_named_cache_group.unwrap_or(false),
      keep_named_chunk: options.keep_named_chunk.unwrap_or(false),
      stages: options.stages.map_or(vec![], |stages| {
        stages
          .into_iter()
          .map(|stage| BetterChunkStage {
            shared: stage.shared.unwrap_or(vec![]),
            entries: stage.entries.map_or(vec![], |entries| match entries {
              Either::A(entries) => entries
                .into_iter()
                .map(|(mut key, mut value)| {
                  if key.parse::<u32>().is_ok() {
                    key = value.expect("value is required");
                    value = None;
                  }
                  (
                    key.clone(),
                    value.as_ref().and_then(|t| {
                      let t = t.split(' ').collect::<Vec<&str>>();
                      let t0 = t.get(0);
                      let t1 = t.get(1).and_then(|t| t.parse::<u32>().ok()).unwrap_or(0) as usize;
                      match t0 {
                        Some(&"preload") => Some((ChunkGroupOrderKey::Preload, t1)),
                        Some(&"prefetch") => Some((ChunkGroupOrderKey::Prefetch, t1)),
                        _ => None,
                      }
                    }),
                  )
                })
                .collect(),
              Either::B(entries) => entries.into_iter().map(|key| (key, None)).collect(),
            }),
            modules: stage.modules.unwrap_or(vec![]),
          })
          .collect()
      }),
      keep_magic_chunks: options.keep_magic_chunks.map_or(vec![], |v| {
        v.into_iter()
          .filter_map(|s| match s.as_str() {
            "preload" => Some(ChunkGroupOrderKey::Preload),
            "prefetch" => Some(ChunkGroupOrderKey::Preload),
            _ => None,
          })
          .collect::<Vec<_>>()
      }),
      remove_duplicate_modules: options.remove_duplicate_modules.unwrap_or(false),
      skip_concat_chunks: options.skip_concat_chunks.unwrap_or(vec![]),
      concat_small_chunks: options.concat_small_chunks.unwrap_or(false),
      concat_unrelated_chunks: options.concat_unrelated_chunks.unwrap_or(false),
      split_big_chunks: options.split_big_chunks.unwrap_or(false),
      split_chunk_sizes: options.split_chunk_sizes.map(|c| vec_to_tuple(c)),
      concat_chunk_sizes: options.concat_chunk_sizes.map(|c| vec_to_tuple(c)),
    }
  }
}
