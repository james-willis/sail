// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! A fair, spill-aware memory pool for Sedona spatial workloads.
//!
//! Copied from apache/sedona-db `rust/sedona/src/memory_pool.rs` at revision
//! `d18b1dee67d2a76b13252bbe24001e335deb5d94` (Apache-2.0), the same revision
//! this fork pins the `sedona-*` crates to. Copying the module (instead of
//! depending on the top-level `sedona` crate) avoids pulling in the whole
//! SedonaDB context stack just for the pool.
//!
//! One deliberate divergence from the copied implementation: spillable
//! consumers are limited by the aggregate spillable budget instead of a static
//! `1/num_spill` per-consumer share. See the comment in
//! [`SedonaFairSpillPool::try_grow`] for the rationale and measurements.

use datafusion::execution::memory_pool::{
    MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation,
};
use datafusion_common::{resources_datafusion_err, DataFusionError, Result};
use parking_lot::Mutex;

pub const DEFAULT_UNSPILLABLE_RESERVE_RATIO: f64 = 0.2;

/// A [`MemoryPool`] implementation similar to DataFusion's [`datafusion::execution::memory_pool::FairSpillPool`],
/// but with the following changes:
///
/// Spillable and non-spillable operators use logically separate portions of the memory pool,
/// controlled by `unspillable_reserve_ratio`, instead of sharing a single pool as in
/// DataFusion's default FairSpillPool, which can lead to the following issue:
/// spillable consumers could potentially exhaust all available memory, preventing unspillable
/// operations from acquiring necessary resources. This behavior is tracked in DataFusion issue
/// <https://github.com/apache/datafusion/issues/17334>. In the context of Sedona, a typical example
/// is a `sedona_spatial_join::exec::SpatialJoinExec` operator with an auto inserted
/// [`datafusion::physical_plan::repartition::RepartitionExec`] for the probe side. The Merge
/// consumer of [`datafusion::physical_plan::repartition::RepartitionExec`] is unspillable, while
/// the `sedona_spatial_join::exec::SpatialJoinExec` is spillable.
/// `sedona_spatial_join::exec::SpatialJoinExec` could consume all memory, resulting in a reservation
/// failure of [`datafusion::physical_plan::repartition::RepartitionExec`].
///
/// By reserving a configurable fraction of the total memory pool specifically for unspillable
/// allocations (defined by `unspillable_reserve_ratio`), this pool ensures that critical
/// non-spillable operations can proceed even under heavy memory pressure from spillable operators.
///
/// Additionally, unlike both DataFusion's FairSpillPool and the copied SedonaDB implementation,
/// spillable consumers share the spillable budget in aggregate instead of being individually
/// capped at `1/num_spill` of it; see [`SedonaFairSpillPool::try_grow`] for the rationale.
#[derive(Debug)]
pub struct SedonaFairSpillPool {
    /// The total memory limit
    pool_size: usize,
    /// The fraction of memory reserved for unspillable consumers (0.0 - 1.0)
    unspillable_reserve_ratio: f64,

    state: Mutex<FairSpillPoolState>,
}

#[derive(Debug)]
struct FairSpillPoolState {
    /// The number of consumers that can spill
    num_spill: usize,

    /// The total amount of memory reserved that can be spilled
    spillable: usize,

    /// The total amount of memory reserved by consumers that cannot spill
    unspillable: usize,
}

impl SedonaFairSpillPool {
    /// Allocate up to `limit` bytes
    pub fn new(pool_size: usize, unspillable_reserve_ratio: f64) -> Self {
        Self {
            pool_size,
            unspillable_reserve_ratio,
            state: Mutex::new(FairSpillPoolState {
                num_spill: 0,
                spillable: 0,
                unspillable: 0,
            }),
        }
    }
}

impl MemoryPool for SedonaFairSpillPool {
    fn register(&self, consumer: &MemoryConsumer) {
        if consumer.can_spill() {
            self.state.lock().num_spill += 1;
        }
    }

    #[expect(clippy::unwrap_used)]
    fn unregister(&self, consumer: &MemoryConsumer) {
        if consumer.can_spill() {
            let mut state = self.state.lock();
            state.num_spill = state.num_spill.checked_sub(1).unwrap();
        }
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        let mut state = self.state.lock();
        match reservation.consumer().can_spill() {
            true => state.spillable += additional,
            false => state.unspillable += additional,
        }
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        let mut state = self.state.lock();
        match reservation.consumer().can_spill() {
            true => state.spillable -= shrink,
            false => state.unspillable -= shrink,
        }
    }

    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        let mut state = self.state.lock();

        // Calculate the amount of memory reserved for unspillable consumers
        let reserved_for_unspillable =
            (self.pool_size as f64 * self.unspillable_reserve_ratio) as usize;

        // The effective unspillable usage is the max of actual usage and the reserved amount
        let effective_unspillable = state.unspillable.max(reserved_for_unspillable);

        // The total amount of memory available to spilling consumers
        let spill_available = self.pool_size.saturating_sub(effective_unspillable);

        match reservation.consumer().can_spill() {
            true => {
                // Divergence from the copied SedonaDB implementation (and DataFusion's
                // FairSpillPool, which both cap each spiller at
                // `spill_available / num_spill`): let any spillable consumer grow as long
                // as the TOTAL spillable usage stays within the spillable budget.
                //
                // The static 1/N division dilutes the per-consumer cap with every
                // *registered* spillable consumer, including idle ones. DataFusion 52
                // registers spillable consumers for every RepartitionExec channel and
                // aggregate stream, so a spatial join build in a typical plan shares its
                // budget with ~3x as many mostly-idle consumers and is pushed to spill
                // while the pool is nearly empty (measured: SpatialBench q10 sf=1 under a
                // 9 GiB pool spilled ~3 GB with peak usage under 2 GiB, because 32
                // registered spillable consumers capped each build partition at 241 MB).
                //
                // Enforcing only the aggregate budget keeps the pool hard-bounded (the
                // whole point of the fair pool) and keeps the unspillable reserve intact,
                // while consumers that hit the wall under real pressure still spill.
                let available = spill_available.saturating_sub(state.spillable);
                if additional > available {
                    return Err(insufficient_capacity_err(
                        reservation,
                        additional,
                        available,
                        state.num_spill,
                        effective_unspillable,
                        spill_available,
                    ));
                }
                state.spillable += additional;
            }
            false => {
                let available = self
                    .pool_size
                    .saturating_sub(state.unspillable + state.spillable);

                if available < additional {
                    return Err(insufficient_capacity_err(
                        reservation,
                        additional,
                        available,
                        state.num_spill,
                        effective_unspillable,
                        spill_available,
                    ));
                }
                state.unspillable += additional;
            }
        }
        Ok(())
    }

    fn reserved(&self) -> usize {
        let state = self.state.lock();
        state.spillable + state.unspillable
    }

    fn memory_limit(&self) -> MemoryLimit {
        MemoryLimit::Finite(self.pool_size)
    }
}

fn insufficient_capacity_err(
    reservation: &MemoryReservation,
    additional: usize,
    available: usize,
    num_spill: usize,
    unspillable: usize,
    spill_available: usize,
) -> DataFusionError {
    resources_datafusion_err!(
        "Failed to allocate additional {} bytes for {} with {} bytes already allocated - maximum available is {} bytes. \
        Current unspillable memory usage: {} bytes, spillable memory available: {} bytes, spillable consumers: {}",
        additional,
        reservation.consumer().name(),
        reservation.size(),
        available,
        unspillable,
        spill_available,
        num_spill
    )
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn test_sedona_fair_spill_pool_reserve() {
        // Pool size 100, 20% reserved for unspillable (20 bytes)
        let pool: Arc<dyn MemoryPool> = Arc::new(SedonaFairSpillPool::new(100, 0.2));

        let spillable_consumer = MemoryConsumer::new("spillable").with_can_spill(true);
        let mut spillable = spillable_consumer.register(&pool);

        let unspillable_consumer = MemoryConsumer::new("unspillable").with_can_spill(false);
        let mut unspillable = unspillable_consumer.register(&pool);

        // Case 1: Spillable cannot eat into reserved memory
        // Available for spillable = 100 - 20 = 80
        spillable.try_grow(80).unwrap();
        assert_eq!(pool.reserved(), 80);

        // Try to grow by 1, should fail because 80 is the limit
        assert!(spillable.try_grow(1).is_err());

        // Case 2: Unspillable can use the reserved memory
        unspillable.try_grow(10).unwrap();
        assert_eq!(pool.reserved(), 90); // 80 (spillable) + 10 (unspillable)

        // Spillable still cannot grow
        assert!(spillable.try_grow(1).is_err());

        // Case 3: Unspillable can grow beyond reserved if space available
        // But currently pool is 90/100.
        // Unspillable wants 15 more (total 25).
        // Available total = 100. Used = 90. Free = 10.
        // Unspillable try_grow(15) -> needs 105 total? No, 90+15 = 105 > 100.
        // Let's shrink spillable first to test "Unspillable eating into Spillable's share"

        spillable.shrink(20); // Spillable = 60. Total = 70.
                              // Unspillable = 10. Reserved = 20.
                              // Effective unspillable = 20. Spill available = 80.
                              // Spillable usage = 60.

        // Unspillable grows by 20. Total unspillable = 30.
        // 30 > 20 (reserved).
        // Total usage = 60 + 30 = 90 <= 100. Should succeed.
        unspillable.try_grow(20).unwrap();
        assert_eq!(pool.reserved(), 90);

        // Now unspillable = 30.
        // Effective unspillable = 30.
        // Spill available = 100 - 30 = 70.
        // Spillable usage = 60.
        // Spillable tries to grow by 11 (60+11=71 > 70). Should fail.
        assert!(spillable.try_grow(11).is_err());
        // Spillable tries to grow by 10 (60+10=70). Should succeed.
        spillable.try_grow(10).unwrap();
        assert_eq!(pool.reserved(), 100);
    }

    #[test]
    fn test_spillers_share_aggregate_budget() {
        // Pool size 100, 0% reserved. Unlike the copied SedonaDB implementation
        // (a strict N-way split per registered spiller), the spillable budget is
        // shared: a consumer may use everything its peers are not using, and the
        // aggregate stays hard-bounded.
        let pool: Arc<dyn MemoryPool> = Arc::new(SedonaFairSpillPool::new(100, 0.0));

        let c1 = MemoryConsumer::new("c1").with_can_spill(true);
        let mut r1 = c1.register(&pool);

        let c2 = MemoryConsumer::new("c2").with_can_spill(true);
        let mut r2 = c2.register(&pool);

        // A single active spiller can use the whole spillable budget even though
        // another (idle) spiller is registered.
        r1.try_grow(100).unwrap();
        assert!(r1.try_grow(1).is_err());

        // The aggregate budget is exhausted, so the second spiller must spill.
        assert!(r2.try_grow(1).is_err());

        // Once the first spiller releases memory, the second can use it.
        r1.shrink(60);
        r2.try_grow(60).unwrap();
        assert_eq!(pool.reserved(), 100);
        assert!(r2.try_grow(1).is_err());
    }

    #[test]
    fn test_spillers_cannot_eat_into_unspillable_reserve() {
        // Pool size 100, 20% reserved for unspillable consumers: the spillable
        // budget is 80 in aggregate, no matter how many spillers are registered.
        let pool: Arc<dyn MemoryPool> = Arc::new(SedonaFairSpillPool::new(100, 0.2));

        let c1 = MemoryConsumer::new("c1").with_can_spill(true);
        let mut r1 = c1.register(&pool);

        let c2 = MemoryConsumer::new("c2").with_can_spill(true);
        let mut r2 = c2.register(&pool);

        r1.try_grow(50).unwrap();
        r2.try_grow(30).unwrap();
        assert!(r1.try_grow(1).is_err());
        assert!(r2.try_grow(1).is_err());
        assert_eq!(pool.reserved(), 80);
    }
}
