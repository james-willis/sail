use std::num::NonZeroUsize;
use std::sync::Arc;

use datafusion::execution::cache::cache_manager::{
    CacheManagerConfig, FileMetadataCache, FileStatisticsCache, ListFilesCache,
};
use datafusion::execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryPool, TrackConsumersPool, UnboundedMemoryPool,
};
use datafusion::execution::runtime_env::{RuntimeEnv, RuntimeEnvBuilder};
use datafusion::execution::DiskManager;
use datafusion_common::Result;
use log::{debug, info, warn};
use sail_cache::file_listing_cache::MokaFileListingCache;
use sail_cache::file_metadata_cache::MokaFileMetadataCache;
use sail_cache::file_statistics_cache::MokaFileStatisticsCache;
use sail_common::config::{
    AppConfig, CacheType, FairMemoryPoolConfig, GreedyMemoryPoolConfig, MemoryPoolConfig,
};
use sail_common::runtime::RuntimeHandle;
use sail_object_store::DynamicObjectStoreRegistry;

use crate::memory_pool::{SedonaFairSpillPool, DEFAULT_UNSPILLABLE_RESERVE_RATIO};
use crate::system_memory::available_memory;

/// The fraction of available memory the auto-sized fair pool claims,
/// matching SedonaDB's default of 75% of RAM.
///
/// The pool bounds only ACCOUNTED allocations, so this was temporarily
/// lowered to 60% while the fair pool's `honest` sharing strategy was the
/// default (a single query's anon footprint could approach the full pool
/// size). With the default back to `diluted` sharing - which pushes large
/// consumers to spill early and keeps the anon footprint far below the pool
/// size, exactly like SedonaDB - the generous sizing is safe again.
/// Explicitly configured sizes are always honored unchanged.
const AUTO_MEMORY_POOL_FRACTION: f64 = 0.75;

/// The fair pool size used when nothing about the machine can be detected;
/// matches the previous fixed default of the `fair` pool (64 GiB).
const FALLBACK_MEMORY_POOL_SIZE: usize = 64 * 1024 * 1024 * 1024;

/// The number of top memory consumers reported in "resources exhausted"
/// errors from the fair pool.
const TRACK_CONSUMERS: NonZeroUsize = NonZeroUsize::MIN.saturating_add(9);

/// The pool size for the `fair` memory pool: the configured size if set, or
/// 75% of the available memory (cgroup limit or total system RAM) when the
/// configured size is `0` (auto).
pub fn resolve_fair_pool_size(configured_max_size: usize) -> usize {
    if configured_max_size != 0 {
        return configured_max_size;
    }
    match available_memory() {
        Some(available) => (available as f64 * AUTO_MEMORY_POOL_FRACTION) as usize,
        None => {
            warn!(
                "cannot detect available memory; using the fallback fair memory pool size of {FALLBACK_MEMORY_POOL_SIZE} bytes"
            );
            FALLBACK_MEMORY_POOL_SIZE
        }
    }
}

/// The effective memory pool limit in bytes implied by the application
/// configuration, or `None` for the unbounded pool.
pub fn memory_pool_limit(config: &MemoryPoolConfig) -> Option<usize> {
    match config {
        MemoryPoolConfig::Unbounded => None,
        MemoryPoolConfig::Greedy(GreedyMemoryPoolConfig { max_size }) => Some(*max_size),
        MemoryPoolConfig::Fair(FairMemoryPoolConfig { max_size, .. }) => {
            Some(resolve_fair_pool_size(*max_size))
        }
    }
}

pub struct RuntimeEnvFactory {
    config: Arc<AppConfig>,
    runtime: RuntimeHandle,
    global_file_listing_cache: Option<Arc<dyn ListFilesCache>>,
    global_file_statistics_cache: Option<Arc<dyn FileStatisticsCache>>,
    global_file_metadata_cache: Option<Arc<MokaFileMetadataCache>>,
}

impl RuntimeEnvFactory {
    pub fn new(config: Arc<AppConfig>, runtime: RuntimeHandle) -> Self {
        Self {
            config,
            runtime,
            global_file_listing_cache: None,
            global_file_statistics_cache: None,
            global_file_metadata_cache: None,
        }
    }

    pub fn create<M>(&mut self, mutator: M) -> Result<Arc<RuntimeEnv>>
    where
        M: FnOnce(RuntimeEnvBuilder) -> Result<RuntimeEnvBuilder>,
    {
        let registry = DynamicObjectStoreRegistry::new(self.runtime.clone());
        let cache_config = CacheManagerConfig::default()
            .with_files_statistics_cache(Some(self.create_file_statistics_cache()))
            .with_list_files_cache(Some(self.create_file_listing_cache()))
            .with_file_metadata_cache(Some(self.create_file_metadata_cache()));
        let builder = RuntimeEnvBuilder::default()
            .with_object_store_registry(Arc::new(registry))
            .with_cache_manager(cache_config)
            .with_memory_pool(self.create_memory_pool())
            .with_disk_manager_builder(self.create_disk_manager_builder());
        let builder = mutator(builder)?;
        Ok(Arc::new(builder.build()?))
    }

    fn create_memory_pool(&self) -> Arc<dyn MemoryPool> {
        match self.config.runtime.memory_pool {
            MemoryPoolConfig::Unbounded => {
                info!("using the unbounded memory pool");
                Arc::new(UnboundedMemoryPool::default())
            }
            MemoryPoolConfig::Greedy(GreedyMemoryPoolConfig { max_size }) => {
                info!("using the greedy memory pool with a limit of {max_size} bytes");
                Arc::new(GreedyMemoryPool::new(max_size))
            }
            MemoryPoolConfig::Fair(FairMemoryPoolConfig {
                max_size,
                sharing_strategy,
            }) => {
                let pool_size = resolve_fair_pool_size(max_size);
                info!(
                    "using the fair memory pool with a limit of {pool_size} bytes{} and {sharing_strategy:?} spillable sharing",
                    if max_size == 0 {
                        " (auto-sized to 75% of available memory)"
                    } else {
                        ""
                    }
                );
                // The Sedona fork of DataFusion's `FairSpillPool` reserves a
                // fraction of the pool for unspillable consumers, so a
                // spillable spatial join cannot starve the merge consumer of
                // an auto-inserted `RepartitionExec` (DataFusion issue #17334).
                Arc::new(TrackConsumersPool::new(
                    SedonaFairSpillPool::new_with_strategy(
                        pool_size,
                        DEFAULT_UNSPILLABLE_RESERVE_RATIO,
                        sharing_strategy,
                    ),
                    TRACK_CONSUMERS,
                ))
            }
        }
    }

    fn create_disk_manager_builder(&self) -> DiskManagerBuilder {
        let max_size = self.config.runtime.temporary_files.max_size;
        let paths = self.config.runtime.temporary_files.paths.as_slice();

        let mut builder = DiskManager::builder();
        builder.set_max_temp_directory_size(max_size as u64);
        if max_size == 0 {
            builder.set_mode(DiskManagerMode::Disabled);
        } else if paths.is_empty() {
            builder.set_mode(DiskManagerMode::OsTmpDirectory);
        } else {
            let paths = paths.iter().map(|x| x.into()).collect();
            builder.set_mode(DiskManagerMode::Directories(paths));
        }
        builder
    }

    fn create_file_statistics_cache(&mut self) -> Arc<dyn FileStatisticsCache> {
        let ttl = self.config.parquet.file_statistics_cache.ttl;
        let max_entries = self.config.parquet.file_statistics_cache.max_entries;
        match &self.config.parquet.file_statistics_cache.r#type {
            CacheType::None => {
                debug!("Not using file statistics cache");
                Arc::new(MokaFileStatisticsCache::new(ttl, Some(0)))
            }
            CacheType::Global => {
                debug!("Using global file statistics cache");
                self.global_file_statistics_cache
                    .get_or_insert_with(|| {
                        Arc::new(MokaFileStatisticsCache::new(ttl, max_entries))
                            as Arc<dyn FileStatisticsCache>
                    })
                    .clone()
            }
            CacheType::Session => {
                debug!("Using session file statistics cache");
                Arc::new(MokaFileStatisticsCache::new(ttl, max_entries))
            }
        }
    }

    fn create_file_listing_cache(&mut self) -> Arc<dyn ListFilesCache> {
        let ttl = self.config.execution.file_listing_cache.ttl;
        let max_entries = self.config.execution.file_listing_cache.max_entries;
        match &self.config.execution.file_listing_cache.r#type {
            CacheType::None => {
                debug!("Not using file listing cache");
                Arc::new(MokaFileListingCache::new(ttl, Some(0)))
            }
            CacheType::Global => {
                debug!("Using global file listing cache");
                self.global_file_listing_cache
                    .get_or_insert_with(|| {
                        Arc::new(MokaFileListingCache::new(ttl, max_entries))
                            as Arc<dyn ListFilesCache>
                    })
                    .clone()
            }
            CacheType::Session => {
                debug!("Using session file listing cache");
                Arc::new(MokaFileListingCache::new(ttl, max_entries))
            }
        }
    }

    fn create_file_metadata_cache(&mut self) -> Arc<dyn FileMetadataCache> {
        let ttl = self.config.parquet.file_metadata_cache.ttl;
        let size_limit = self.config.parquet.file_metadata_cache.size_limit;
        match self.config.parquet.file_metadata_cache.r#type {
            CacheType::None => {
                debug!("Not using file metadata cache");
                Arc::new(MokaFileMetadataCache::new(ttl, Some(0)))
            }
            CacheType::Global => {
                debug!("Using global file metadata cache");
                self.global_file_metadata_cache
                    .get_or_insert_with(|| Arc::new(MokaFileMetadataCache::new(ttl, size_limit)))
                    .clone()
            }
            CacheType::Session => {
                debug!("Using session file metadata cache");
                Arc::new(MokaFileMetadataCache::new(ttl, size_limit))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_fair_pool_size_explicit() {
        assert_eq!(resolve_fair_pool_size(1024), 1024);
        assert_eq!(
            resolve_fair_pool_size(12 * 1024 * 1024 * 1024),
            12 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn test_resolve_fair_pool_size_auto() {
        let size = resolve_fair_pool_size(0);
        assert!(size > 0);
        if let Some(available) = available_memory() {
            assert_eq!(
                size,
                (available as f64 * AUTO_MEMORY_POOL_FRACTION) as usize
            );
            assert!(size < available as usize);
        } else {
            assert_eq!(size, FALLBACK_MEMORY_POOL_SIZE);
        }
    }

    #[test]
    fn test_memory_pool_limit_mapping() {
        assert_eq!(memory_pool_limit(&MemoryPoolConfig::Unbounded), None);
        assert_eq!(
            memory_pool_limit(&MemoryPoolConfig::Greedy(GreedyMemoryPoolConfig {
                max_size: 123,
            })),
            Some(123)
        );
        assert_eq!(
            memory_pool_limit(&MemoryPoolConfig::Fair(FairMemoryPoolConfig {
                max_size: 456,
                sharing_strategy: Default::default(),
            })),
            Some(456)
        );
        // An auto-sized fair pool still reports a finite limit.
        let auto = memory_pool_limit(&MemoryPoolConfig::Fair(FairMemoryPoolConfig {
            max_size: 0,
            sharing_strategy: Default::default(),
        }));
        assert!(auto.is_some_and(|size| size > 0));
    }
}
