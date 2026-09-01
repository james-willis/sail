//! End-to-end SQL tests for the geoparquet table format: geometry columns
//! read from GeoParquet files must keep their geoarrow.wkb extension typing
//! through plan resolution, optimization, physical planning, and execution,
//! so that spatial function kernels match consistently.

#![expect(clippy::panic)]

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{Array, BinaryArray, Int64Array, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::parquet::file::metadata::KeyValue;
use datafusion::parquet::file::properties::WriterProperties;
use datafusion::physical_plan::collect;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::{DataFusionError, Result};
use sail_catalog::manager::{CatalogManager, CatalogManagerOptions};
use sail_catalog::provider::CatalogProvider;
use sail_catalog_memory::MemoryCatalogProvider;
use sail_common_datafusion::catalog::display::DefaultCatalogDisplay;
use sail_common_datafusion::datasource::TableFormatRegistry;
use sail_common_datafusion::session::plan::PlanService;
use sail_data_source::formats::geoparquet::GeoParquetTableFormat;
use sail_plan::catalog::SparkCatalogObjectDisplay;
use sail_plan::config::PlanConfig;
use sail_plan::execute_logical_plan;
use sail_plan::formatter::SparkPlanFormatter;
use sail_plan::resolver::plan::NamedPlan;
use sail_plan::resolver::PlanResolver;
use sail_session::optimizer::{default_analyzer_rules, default_optimizer_rules};
use sail_session::planner::new_query_planner;
use sail_sql_analyzer::parser::parse_one_statement;
use sail_sql_analyzer::statement::from_ast_statement;

/// A little-endian WKB point.
fn wkb_point(x: f64, y: f64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(21);
    buf.push(1u8);
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&x.to_le_bytes());
    buf.extend_from_slice(&y.to_le_bytes());
    buf
}

/// Write a GeoParquet file with a WKB `geometry` column and `geo` footer metadata.
fn write_geoparquet_file(path: &std::path::Path, points: &[(f64, f64)]) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("geometry", DataType::Binary, true),
    ]));
    let wkb: Vec<Vec<u8>> = points.iter().map(|(x, y)| wkb_point(*x, *y)).collect();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(
                (1..=points.len() as i64).collect::<Vec<_>>(),
            )),
            Arc::new(BinaryArray::from_vec(
                wkb.iter().map(|b| b.as_slice()).collect(),
            )),
        ],
    )?;
    let geo = r#"{"version":"1.0.0","primary_column":"geometry","columns":{"geometry":{"encoding":"WKB","geometry_types":["Point"]}}}"#;
    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue::new(
            "geo".to_string(),
            geo.to_string(),
        )]))
        .build();
    let file = std::fs::File::create(path).map_err(|e| DataFusionError::External(e.into()))?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Build a session that mirrors the server session for SQL resolution:
/// table format registry, catalog manager, plan service, and the same
/// analyzer/optimizer/physical-optimizer rules and query planner.
fn create_session() -> Result<SessionContext> {
    let registry = Arc::new(TableFormatRegistry::new());
    registry.register(Arc::new(GeoParquetTableFormat::default()))?;

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
    let config = sail_sedona::add_sedona_option_extension(config);

    let builder = SessionStateBuilder::new()
        .with_config(config)
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

/// Resolve and execute a SQL query the way the server does: parse to the plan
/// spec, resolve with the plan resolver, optimize, plan physically, execute.
async fn run_sql(ctx: &SessionContext, sql: &str) -> Result<Vec<RecordBatch>> {
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
    collect(plan, ctx.task_ctx()).await
}

fn string_column(batches: &[RecordBatch], index: usize) -> Vec<String> {
    let mut values = vec![];
    for batch in batches {
        let column = batch.column(index);
        let column = column
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap_or_else(|| panic!("expected string column, got {:?}", column.data_type()));
        for i in 0..column.len() {
            values.push(column.value(i).to_string());
        }
    }
    values
}

#[tokio::test]
async fn test_geoparquet_sql_spatial_function_typing() -> Result<()> {
    let ctx = create_session()?;

    // Two file layouts: distinct geometry values, and a constant geometry
    // column. The latter has exact min == max column statistics, which makes
    // DataFusion's Parquet opener try to replace the column reference with a
    // plain literal (losing extension typing without the statistics handling
    // in the geoparquet table format).
    for (name, points) in [
        ("distinct", [(1.0, 2.0), (3.0, 4.0)]),
        ("constant", [(1.0, 2.0), (1.0, 2.0)]),
    ] {
        let dir = tempfile::tempdir().map_err(|e| DataFusionError::External(e.into()))?;
        let path = dir.path().join("part-0.parquet");
        write_geoparquet_file(&path, &points)?;
        let table = format!("geoparquet.`{}`", dir.path().to_string_lossy());

        // ST_AsText on the geometry column directly: the column must resolve
        // as geometry (WKB with geoarrow.wkb extension metadata) all the way
        // through execution so the kernel matches.
        let sql = format!("SELECT id, ST_AsText(geometry) FROM {table} ORDER BY id");
        let batches = run_sql(&ctx, &sql)
            .await
            .unwrap_or_else(|e| panic!("ST_AsText({name}) failed: {e}"));
        assert_eq!(
            string_column(&batches, 1),
            vec![
                format!("POINT({} {})", points[0].0, points[0].1),
                format!("POINT({} {})", points[1].0, points[1].1),
            ],
            "unexpected ST_AsText output for {name} geometry"
        );

        // Spark clients treat geometry columns as WKB bytes, so functions
        // that parse raw WKB must also accept the geometry-typed column.
        let sql =
            format!("SELECT ST_AsText(ST_GeomFromWKB(geometry)), id FROM {table} ORDER BY id");
        let batches = run_sql(&ctx, &sql)
            .await
            .unwrap_or_else(|e| panic!("ST_GeomFromWKB({name}) failed: {e}"));
        assert_eq!(
            string_column(&batches, 0),
            vec![
                format!("POINT({} {})", points[0].0, points[0].1),
                format!("POINT({} {})", points[1].0, points[1].1),
            ],
            "unexpected ST_GeomFromWKB output for {name} geometry"
        );
    }

    Ok(())
}
