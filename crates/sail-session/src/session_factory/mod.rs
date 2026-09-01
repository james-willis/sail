mod server;
mod worker;

use datafusion::common::Result;
use datafusion::execution::memory_pool::MemoryLimit;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::prelude::{SessionConfig, SessionContext};
use log::info;
use sedona_common::option::SedonaOptions;
pub use server::{ServerSessionFactory, ServerSessionInfo, ServerSessionMutator};
pub use worker::WorkerSessionFactory;

pub trait SessionFactory<I>: Send {
    /// Create a DataFusion [`SessionContext`].
    /// This method takes `&mut self` so that the factory can maintain internal state if needed.
    /// This method takes an opaque parameter of type `I` for session-specific information.
    fn create(&mut self, info: I) -> Result<SessionContext>;
}

/// Batches larger than this fraction of the per-partition memory limit are
/// broken into smaller batches before being written to spill files, to avoid
/// overshooting the memory limit when reading super large spilled batches
/// back. This mirrors SedonaDB's context setup (`rust/sedona/src/context.rs`):
/// 5% of the per-partition memory limit, with a 10 MB minimum.
const SPILLED_BATCH_THRESHOLD_PERCENT_DIVISOR: usize = 20; // 5% == 1 / 20
const MIN_SPILLED_BATCH_IN_MEMORY_THRESHOLD_BYTES: usize = 10 * 1024 * 1024; // 10MB

/// The spatial-join spilled-batch in-memory size threshold for a given memory
/// pool limit and number of target partitions.
pub(crate) fn spilled_batch_in_memory_size_threshold(
    memory_limit: usize,
    target_partitions: usize,
) -> usize {
    let per_partition_memory_limit = memory_limit.div_ceil(target_partitions.max(1));
    per_partition_memory_limit
        .div_ceil(SPILLED_BATCH_THRESHOLD_PERCENT_DIVISOR)
        .max(MIN_SPILLED_BATCH_IN_MEMORY_THRESHOLD_BYTES)
}

/// Derive `spatial_join.spilled_batch_in_memory_size_threshold` from the
/// memory pool limit of the runtime environment, like SedonaDB does when its
/// context runs with a bounded memory pool. This is a no-op when the memory
/// pool is unbounded or `SedonaOptions` are not registered on the config.
pub(crate) fn configure_spatial_join_spill_threshold(
    config: &mut SessionConfig,
    runtime: &RuntimeEnv,
) {
    let MemoryLimit::Finite(memory_limit) = runtime.memory_pool.memory_limit() else {
        return;
    };
    let target_partitions = config.options().execution.target_partitions;
    if let Some(options) = config.options_mut().extensions.get_mut::<SedonaOptions>() {
        let threshold = spilled_batch_in_memory_size_threshold(memory_limit, target_partitions);
        options.spatial_join.spilled_batch_in_memory_size_threshold = threshold;
        info!(
            "setting the spatial join spilled batch in-memory size threshold to {threshold} bytes \
            (memory pool limit {memory_limit} bytes, {target_partitions} target partitions)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spilled_batch_in_memory_size_threshold() {
        const MIB: usize = 1024 * 1024;
        const GIB: usize = 1024 * MIB;
        // 9 GiB pool over 8 partitions: 1.125 GiB per partition, 5% = 57.6 MiB.
        assert_eq!(
            spilled_batch_in_memory_size_threshold(9 * GIB, 8),
            (9 * GIB).div_ceil(8).div_ceil(20)
        );
        // A small pool clamps to the 10 MB minimum.
        assert_eq!(
            spilled_batch_in_memory_size_threshold(64 * MIB, 8),
            10 * MIB
        );
        // Zero partitions do not divide by zero.
        assert_eq!(
            spilled_batch_in_memory_size_threshold(GIB, 0),
            GIB.div_ceil(20)
        );
    }
}
