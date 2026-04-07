// Licensed under the Apache License, Version 2.0.
// This crate bridges SedonaDB's geospatial function crates into Sail's
// function registration system. It assembles SedonaDB's FunctionSet and
// provides lookup APIs for Sail's plan resolver and execution codec.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::prelude::SessionContext;
use datafusion_common::Result;
use datafusion_expr::{AggregateUDF, AggregateUDFImpl, ScalarUDF, ScalarUDFImpl};
use lazy_static::lazy_static;
use sedona_expr::function_set::FunctionSet;

/// Build the complete SedonaDB function set, mirroring the registration logic
/// from SedonaContext::new_from_context() in sedona-db/rust/sedona/src/context.rs.
pub fn build_sedona_function_set() -> Result<FunctionSet> {
    let mut functions = FunctionSet::new();

    // Always register core functions (~50 scalar + 3 aggregate)
    functions.merge(sedona_functions::register::default_function_set());

    // Register GEOS kernels (overlay, predicates, topology)
    #[cfg(feature = "geos")]
    {
        for (name, kernels) in sedona_geos::register::scalar_kernels() {
            functions.add_scalar_udf_impl(name, kernels)?;
        }
        for (name, kernels) in sedona_geos::register::aggregate_kernels() {
            functions.add_aggregate_udf_kernel(name, kernels)?;
        }
    }

    // Register geo-rs kernels (area, buffer, centroid, distance, etc.)
    #[cfg(feature = "geo")]
    {
        for (name, kernels) in sedona_geo::register::scalar_kernels() {
            functions.add_scalar_udf_impl(name, kernels)?;
        }
        for (name, kernels) in sedona_geo::register::aggregate_kernels() {
            functions.add_aggregate_udf_kernel(name, kernels)?;
        }
    }

    // Register TinyGEO kernels (fast spatial predicates)
    #[cfg(feature = "tg")]
    {
        for (name, kernels) in sedona_tg::register::scalar_kernels() {
            functions.add_scalar_udf_impl(name, kernels)?;
        }
    }

    // Register PROJ kernels (ST_Transform)
    #[cfg(feature = "proj")]
    {
        for (name, kernel) in sedona_proj::register::scalar_kernels() {
            functions.add_scalar_udf_impl(name, kernel)?;
        }
    }

    // Register raster functions
    #[cfg(feature = "raster")]
    functions.merge(sedona_raster_functions::register::default_function_set());

    Ok(functions)
}

/// Register all SedonaDB functions on a DataFusion SessionContext.
pub fn register_sedona_udfs(ctx: &SessionContext) -> Result<()> {
    let functions = build_sedona_function_set()?;
    for udf in functions.scalar_udfs() {
        ctx.register_udf(ScalarUDF::from(udf.clone()));
    }
    for udaf in functions.aggregate_udfs() {
        ctx.register_udaf(AggregateUDF::from(udaf.clone()));
    }
    Ok(())
}

lazy_static! {
    static ref SEDONA_SCALAR_REGISTRY: HashMap<String, Arc<ScalarUDF>> = {
        let fs = build_sedona_function_set().expect("failed to build sedona function set");
        fs.scalar_udfs()
            .map(|udf| {
                let name = udf.name().to_string();
                (name, Arc::new(ScalarUDF::from(udf.clone())))
            })
            .collect()
    };
    static ref SEDONA_AGGREGATE_REGISTRY: HashMap<String, Arc<AggregateUDF>> = {
        let fs = build_sedona_function_set().expect("failed to build sedona function set");
        fs.aggregate_udfs()
            .map(|udaf| {
                let name = udaf.name().to_string();
                (name, Arc::new(AggregateUDF::from(udaf.clone())))
            })
            .collect()
    };
}

/// Look up a SedonaDB scalar UDF by name (for execution codec).
pub fn get_sedona_scalar_udf(name: &str) -> Option<Arc<ScalarUDF>> {
    SEDONA_SCALAR_REGISTRY.get(name).cloned()
}

/// Look up a SedonaDB aggregate UDF by name (for execution codec).
pub fn get_sedona_aggregate_udf(name: &str) -> Option<Arc<AggregateUDF>> {
    SEDONA_AGGREGATE_REGISTRY.get(name).cloned()
}

/// Get all registered SedonaDB scalar function names and their UDFs.
/// Used by sail-plan to build the planning-time function registry.
pub fn sedona_scalar_udfs() -> impl Iterator<Item = (&'static str, Arc<ScalarUDF>)> {
    SEDONA_SCALAR_REGISTRY
        .iter()
        .map(|(name, udf)| (name.as_str(), udf.clone()))
}

/// Get all registered SedonaDB aggregate function names and their UDAFs.
pub fn sedona_aggregate_udfs() -> impl Iterator<Item = (&'static str, Arc<AggregateUDF>)> {
    SEDONA_AGGREGATE_REGISTRY
        .iter()
        .map(|(name, udaf)| (name.as_str(), udaf.clone()))
}
