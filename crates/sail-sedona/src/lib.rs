// Licensed under the Apache License, Version 2.0.
// This crate bridges SedonaDB's geospatial function crates into Sail's
// function registration system. It assembles SedonaDB's FunctionSet and
// provides lookup APIs for Sail's plan resolver and execution codec.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::prelude::SessionContext;
use datafusion_common::Result;
use datafusion_expr::{AggregateUDF, ScalarUDF};
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

    // ST_Transform kernels are registered by sedona-functions; the `proj`
    // feature supplies the CRS engine behind them (see configure_proj_engine).

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
        let mut registry = HashMap::new();
        for udf in fs.scalar_udfs() {
            let udf = Arc::new(ScalarUDF::from(udf.clone()));
            // Register under the primary name and every alias (e.g.
            // st_geomfromtext -> st_geomfromwkt) so Spark-style names resolve.
            registry.insert(udf.name().to_string(), udf.clone());
            for alias in udf.aliases() {
                registry.entry(alias.clone()).or_insert_with(|| udf.clone());
            }
        }
        registry
    };
    static ref SEDONA_AGGREGATE_REGISTRY: HashMap<String, Arc<AggregateUDF>> = {
        let fs = build_sedona_function_set().expect("failed to build sedona function set");
        let mut registry = HashMap::new();
        for udaf in fs.aggregate_udfs() {
            let udaf = Arc::new(AggregateUDF::from(udaf.clone()));
            registry.insert(udaf.name().to_string(), udaf.clone());
            for alias in udaf.aliases() {
                registry.entry(alias.clone()).or_insert_with(|| udaf.clone());
            }
        }
        registry
    };
}

/// Register SedonaOptions on a SessionConfig, wiring the PROJ-backed CRS
/// engine into the runtime when the `proj` feature is enabled (mirrors
/// sedona-db's own context setup). Without this, ST_Transform sees the
/// DefaultCrsEngine and fails with "no CrsEngine registered" even when the
/// global PROJ engine has been configured.
pub fn add_sedona_option_extension(
    config: datafusion::execution::context::SessionConfig,
) -> datafusion::execution::context::SessionConfig {
    use sedona_common::option::SedonaOptions;

    #[allow(unused_mut)]
    let mut config = config.with_option_extension(SedonaOptions::default());
    #[cfg(feature = "proj")]
    if let Some(opts) = config
        .options_mut()
        .extensions
        .get_mut::<SedonaOptions>()
    {
        opts.runtime = opts
            .runtime
            .with_crs_engine(std::sync::Arc::new(sedona_proj::transform::LazyProjEngine));
    }
    config
}

/// Configure the global PROJ CRS engine used by ST_Transform.
///
/// libproj is loaded dynamically at runtime; callers pass the path to a
/// PROJ shared library (e.g. the one bundled inside the pyproj wheel),
/// its proj.db, and a resource search path. Any argument may be None to
/// let sedona-proj use its platform defaults.
#[cfg(feature = "proj")]
pub fn configure_proj_engine(
    shared_library_path: Option<String>,
    database_path: Option<String>,
    search_path: Option<String>,
) -> Result<()> {
    use sedona_proj::register::{configure_global_proj_engine, ProjCrsEngineBuilder};

    let mut builder = ProjCrsEngineBuilder::default();
    if let Some(path) = shared_library_path {
        builder = builder.with_shared_library(path.into());
    }
    if let Some(path) = database_path {
        builder = builder.with_database_path(path.into());
    }
    if let Some(path) = search_path {
        builder = builder.with_search_paths(vec![path.into()]);
    }
    configure_global_proj_engine(builder)
        .map_err(|e| datafusion_common::DataFusionError::External(Box::new(e)))
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
