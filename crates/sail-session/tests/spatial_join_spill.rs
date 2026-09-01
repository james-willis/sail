//! End-to-end test for spatial-join spilling under a bounded memory pool:
//! a session with a small `SedonaFairSpillPool` and an enabled disk manager
//! must complete a spatial join by spilling to disk instead of failing with
//! a memory reservation error.

#![expect(clippy::panic, clippy::unwrap_used)]

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{Array, BinaryArray, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::physical_plan::{collect, ExecutionPlan};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::{DataFusionError, Result};
use sail_catalog::manager::{CatalogManager, CatalogManagerOptions};
use sail_catalog::provider::CatalogProvider;
use sail_catalog_memory::MemoryCatalogProvider;
use sail_common_datafusion::catalog::display::DefaultCatalogDisplay;
use sail_common_datafusion::datasource::TableFormatRegistry;
use sail_common_datafusion::session::plan::PlanService;
use sail_data_source::formats::parquet::ParquetTableFormat;
use sail_plan::catalog::SparkCatalogObjectDisplay;
use sail_plan::config::PlanConfig;
use sail_plan::execute_logical_plan;
use sail_plan::formatter::SparkPlanFormatter;
use sail_plan::resolver::plan::NamedPlan;
use sail_plan::resolver::PlanResolver;
use sail_session::memory_pool::SedonaFairSpillPool;
use sail_session::optimizer::{default_analyzer_rules, default_optimizer_rules};
use sail_session::planner::new_query_planner;
use sail_sql_analyzer::parser::parse_one_statement;
use sail_sql_analyzer::statement::from_ast_statement;
use sedona_common::option::SedonaOptions;

/// A little-endian WKB point.
fn wkb_point(x: f64, y: f64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(21);
    buf.push(1u8);
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&x.to_le_bytes());
    buf.extend_from_slice(&y.to_le_bytes());
    buf
}

/// A little-endian WKB polygon with a single rectangular ring.
fn wkb_rectangle(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(1u8);
    buf.extend_from_slice(&3u32.to_le_bytes()); // polygon
    buf.extend_from_slice(&1u32.to_le_bytes()); // one ring
    buf.extend_from_slice(&5u32.to_le_bytes()); // five points (closed)
    for (x, y) in [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
        (x_min, y_min),
    ] {
        buf.extend_from_slice(&x.to_le_bytes());
        buf.extend_from_slice(&y.to_le_bytes());
    }
    buf
}

/// Write a Parquet file with an `id` column and a WKB `geom` column.
fn write_parquet_file(path: &std::path::Path, geoms: &[Vec<u8>]) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("geom", DataType::Binary, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(
                (1..=geoms.len() as i64).collect::<Vec<_>>(),
            )),
            Arc::new(BinaryArray::from_vec(
                geoms.iter().map(|b| b.as_slice()).collect(),
            )),
        ],
    )?;
    let file = std::fs::File::create(path).map_err(|e| DataFusionError::External(e.into()))?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Build a session that mirrors the server session for SQL resolution, with a
/// small bounded `SedonaFairSpillPool` and forced spatial-join spilling.
fn create_session(memory_pool_size: usize) -> Result<SessionContext> {
    let registry = Arc::new(TableFormatRegistry::new());
    registry.register(Arc::new(ParquetTableFormat::default()))?;

    let catalog_manager = CatalogManager::try_new(CatalogManagerOptions {
        catalogs: HashMap::from([(
            "sail".to_string(),
            Arc::new(MemoryCatalogProvider::new(
                "sail".to_string(),
                vec![Arc::from("default")]
                    .try_into()
                    .map_err(|e| DataFusionError::External(Box::new(e)))?,
                None,
            )) as Arc<dyn CatalogProvider>,
        )]),
        default_catalog: "sail".to_string(),
        default_database: vec!["default".to_string()],
        global_temporary_database: vec!["global_temp".to_string()],
    })
    .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let plan_service = PlanService::new(
        Box::new(DefaultCatalogDisplay::<SparkCatalogObjectDisplay>::default()),
        Box::new(SparkPlanFormatter),
    );

    let config = SessionConfig::new()
        .with_create_default_catalog_and_schema(false)
        .with_information_schema(false)
        .with_extension(registry)
        .with_extension(Arc::new(catalog_manager))
        .with_extension(Arc::new(plan_service));
    let mut config = sail_sedona::add_sedona_option_extension(config);
    {
        let options = config
            .options_mut()
            .extensions
            .get_mut::<SedonaOptions>()
            .unwrap();
        // Make spilling deterministic regardless of how much memory the small
        // join actually reserves.
        options.spatial_join.debug.force_spill = true;
        options.spatial_join.spilled_batch_in_memory_size_threshold = 10 * 1024 * 1024;
    }

    // A bounded fair pool; the default disk manager (OS temporary directory)
    // stays enabled so spilling has somewhere to go.
    let runtime = RuntimeEnvBuilder::default()
        .with_memory_pool(Arc::new(SedonaFairSpillPool::new(memory_pool_size, 0.2)))
        .build_arc()?;

    let builder = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(runtime)
        .with_analyzer_rules(default_analyzer_rules())
        .with_optimizer_rules(default_optimizer_rules())
        .with_physical_optimizer_rules(sail_physical_optimizer::get_physical_optimizers(
            sail_physical_optimizer::PhysicalOptimizerOptions {
                enable_join_reorder: false,
            },
        ))
        .with_query_planner(new_query_planner());
    let builder = sedona_query_planner::optimizer::register_spatial_join_logical_optimizer(builder)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    Ok(SessionContext::new_with_state(builder.build()))
}

/// Resolve and execute a SQL query the way the server does, returning the
/// result batches and the executed physical plan (for metrics inspection).
async fn run_sql(
    ctx: &SessionContext,
    sql: &str,
) -> Result<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>)> {
    let plan = from_ast_statement(
        parse_one_statement(sql).map_err(|e| DataFusionError::External(Box::new(e)))?,
    )
    .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let resolver = PlanResolver::new(
        ctx,
        Arc::new(PlanConfig::new().map_err(|e| DataFusionError::External(Box::new(e)))?),
    );
    let NamedPlan { plan, fields: _ } = resolver
        .resolve_named_plan(plan)
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let df = execute_logical_plan(ctx, plan).await?;
    let (session_state, plan) = df.into_parts();
    let plan = session_state.optimize(&plan)?;
    let plan = session_state
        .query_planner()
        .create_physical_plan(&plan, &session_state)
        .await?;
    let batches = collect(plan.clone(), ctx.task_ctx()).await?;
    Ok((batches, plan))
}

/// The total spill count over all operators in the plan.
fn total_spill_count(plan: &Arc<dyn ExecutionPlan>) -> usize {
    let own = plan
        .metrics()
        .and_then(|metrics| metrics.spill_count())
        .unwrap_or(0);
    own + plan
        .children()
        .iter()
        .map(|child| total_spill_count(child))
        .sum::<usize>()
}

fn single_int64_value(batches: &[RecordBatch]) -> i64 {
    let mut values = vec![];
    for batch in batches {
        let column = batch.column(0);
        let column = column
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap_or_else(|| panic!("expected int64 column, got {:?}", column.data_type()));
        for i in 0..column.len() {
            values.push(column.value(i));
        }
    }
    assert_eq!(values.len(), 1, "expected a single value");
    values[0]
}

#[tokio::test]
async fn test_spatial_join_spills_under_bounded_memory_pool() -> Result<()> {
    // 10 unit grid cells [i, i+1] x [0, 1] and `points_per_cell` points in
    // each cell, so ST_Within(point, cell) matches each point exactly once.
    let cells = 10usize;
    let points_per_cell = 20_000usize;

    let dir = tempfile::tempdir().map_err(|e| DataFusionError::External(e.into()))?;
    let zone_path = dir.path().join("zones.parquet");
    let point_path = dir.path().join("points.parquet");

    let zones: Vec<Vec<u8>> = (0..cells)
        .map(|i| wkb_rectangle(i as f64, 0.0, i as f64 + 1.0, 1.0))
        .collect();
    write_parquet_file(&zone_path, &zones)?;

    let points: Vec<Vec<u8>> = (0..cells * points_per_cell)
        .map(|i| {
            let cell = (i % cells) as f64;
            let offset = ((i / cells) as f64 + 1.0) / (points_per_cell as f64 + 2.0);
            wkb_point(cell + offset, offset)
        })
        .collect();
    write_parquet_file(&point_path, &points)?;

    // 64 MiB pool: small enough that a runaway join would fail, large enough
    // for the scan and aggregation plumbing of this query.
    let ctx = create_session(64 * 1024 * 1024)?;

    let sql = format!(
        "SELECT COUNT(*) FROM parquet.`{}` p JOIN parquet.`{}` z \
        ON ST_Within(ST_GeomFromWKB(p.geom), ST_GeomFromWKB(z.geom))",
        point_path.to_string_lossy(),
        zone_path.to_string_lossy(),
    );
    let (batches, plan) = run_sql(&ctx, &sql).await?;

    assert_eq!(
        single_int64_value(&batches),
        (cells * points_per_cell) as i64,
        "spatial join under a bounded memory pool returned the wrong count"
    );
    assert!(
        total_spill_count(&plan) > 0,
        "expected the spatial join to spill to disk"
    );
    Ok(())
}
