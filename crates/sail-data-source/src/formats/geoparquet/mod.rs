//! GeoParquet table format backed by the `sedona-geoparquet` crate.
//!
//! This adapts SedonaDB's [`GeoParquetFormat`] (a DataFusion [`FileFormat`]
//! wrapping the built-in Parquet format) into Sail's [`ListingFormat`] /
//! `TableFormat` machinery, so that Spark Connect clients can use
//! `spark.read.format("geoparquet")` and `df.write.format("geoparquet")`.
//!
//! On read, the format inspects the `geo` key/value metadata in Parquet file
//! footers and annotates matching columns with the `geoarrow.wkb` Arrow
//! extension type, which SedonaDB's spatial functions and the spatial join
//! planner understand. Plain Parquet files (no `geo` metadata) read fine and
//! behave exactly like the regular Parquet format.
//!
//! All regular Parquet options are supported and resolved through Sail's
//! Parquet option resolution. In addition, the following GeoParquet-specific
//! options are recognized:
//! - `geoparquet_version`: `1.0` (default), `1.1`, `2.0`, or `none` (write)
//! - `overwrite_bbox_columns`: overwrite existing bbox columns (write, 1.1)
//! - `geometry_columns`: JSON metadata overrides for schema inference (read)
//! - `validate`: validate geometry against metadata (read)

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::catalog::Session;
use datafusion::config::ConfigField;
use datafusion_common::parsers::CompressionTypeVariant;
use datafusion_common::Result;
use datafusion_datasource::file_format::FileFormat;
use sedona_geoparquet::format::GeoParquetFormat;
use sedona_geoparquet::options::TableGeoParquetOptions;

use crate::formats::listing::{DefaultSchemaInfer, ListingFormat, ListingTableFormat, SchemaInfer};
use crate::formats::parquet::options::{
    resolve_parquet_read_options, resolve_parquet_write_options,
};

pub type GeoParquetTableFormat = ListingTableFormat<GeoParquetListingFormat>;

/// GeoParquet-specific option keys (must be lowercase).
/// These are applied on top of the regular Parquet options; all other keys go
/// through Sail's Parquet option resolution (which ignores unknown keys).
const GEOPARQUET_OPTION_KEYS: [&str; 4] = [
    "geoparquet_version",
    "overwrite_bbox_columns",
    "geometry_columns",
    "validate",
];

fn apply_geoparquet_options(
    to: &mut TableGeoParquetOptions,
    options: &[HashMap<String, String>],
) -> Result<()> {
    for opts in options {
        for (key, value) in opts {
            let key = key.to_lowercase();
            if GEOPARQUET_OPTION_KEYS.contains(&key.as_str()) && !value.is_empty() {
                to.set(&key, value)?;
            }
        }
    }
    Ok(())
}

#[derive(Debug, Default)]
pub struct GeoParquetListingFormat;

impl ListingFormat for GeoParquetListingFormat {
    fn name(&self) -> &'static str {
        "geoparquet"
    }

    fn create_read_format(
        &self,
        ctx: &dyn Session,
        options: Vec<HashMap<String, String>>,
        _compression: Option<CompressionTypeVariant>,
    ) -> Result<Arc<dyn FileFormat>> {
        let parquet_options = resolve_parquet_read_options(ctx, options.clone())?;
        let mut geo_options = TableGeoParquetOptions::from(parquet_options);
        apply_geoparquet_options(&mut geo_options, &options)?;
        Ok(Arc::new(GeoParquetFormat::new(geo_options)))
    }

    fn create_write_format(
        &self,
        ctx: &dyn Session,
        options: Vec<HashMap<String, String>>,
    ) -> Result<(Arc<dyn FileFormat>, Option<String>)> {
        let parquet_options = resolve_parquet_write_options(ctx, options.clone())?;
        let compression = parquet_options.global.compression.clone();
        let mut geo_options = TableGeoParquetOptions::from(parquet_options);
        apply_geoparquet_options(&mut geo_options, &options)?;
        Ok((Arc::new(GeoParquetFormat::new(geo_options)), compression))
    }

    fn schema_inferrer(&self) -> Arc<dyn SchemaInfer> {
        Arc::new(DefaultSchemaInfer)
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::{BinaryArray, Int64Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::parquet::arrow::ArrowWriter;
    use datafusion::parquet::file::metadata::KeyValue;
    use datafusion::parquet::file::properties::WriterProperties;
    use datafusion::prelude::SessionContext;
    use datafusion_common::{Constraints, DataFusionError};
    use sail_common_datafusion::datasource::{SourceInfo, TableFormat};

    use super::*;

    /// A little-endian WKB point.
    fn wkb_point(x: f64, y: f64) -> Vec<u8> {
        let mut buf = Vec::with_capacity(21);
        buf.push(1u8); // little-endian
        buf.extend_from_slice(&1u32.to_le_bytes()); // geometry type: point
        buf.extend_from_slice(&x.to_le_bytes());
        buf.extend_from_slice(&y.to_le_bytes());
        buf
    }

    /// Write a Parquet file with a WKB `geometry` column, optionally with
    /// GeoParquet `geo` footer metadata.
    fn write_parquet_file(path: &std::path::Path, with_geo_metadata: bool) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("geometry", DataType::Binary, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(BinaryArray::from_vec(vec![
                    &wkb_point(1.0, 2.0),
                    &wkb_point(3.0, 4.0),
                    &wkb_point(5.0, 6.0),
                ])),
            ],
        )?;
        let mut builder = WriterProperties::builder();
        if with_geo_metadata {
            let geo = r#"{"version":"1.0.0","primary_column":"geometry","columns":{"geometry":{"encoding":"WKB","geometry_types":["Point"]}}}"#;
            builder = builder.set_key_value_metadata(Some(vec![KeyValue::new(
                "geo".to_string(),
                geo.to_string(),
            )]));
        }
        let file = std::fs::File::create(path).map_err(|e| DataFusionError::External(e.into()))?;
        let mut writer = ArrowWriter::try_new(file, schema, Some(builder.build()))?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }

    fn source_info(path: &std::path::Path) -> SourceInfo {
        SourceInfo {
            paths: vec![path.to_string_lossy().to_string()],
            schema: None,
            constraints: Constraints::default(),
            partition_by: vec![],
            bucket_by: None,
            sort_order: vec![],
            options: vec![],
        }
    }

    #[tokio::test]
    async fn test_read_geoparquet_infers_geoarrow_extension() -> Result<()> {
        let dir = tempfile::tempdir().map_err(|e| DataFusionError::External(e.into()))?;
        let path = dir.path().join("part-0.parquet");
        write_parquet_file(&path, true)?;

        let ctx = SessionContext::new();
        let state = ctx.state();
        let format = GeoParquetTableFormat::default();
        let provider = format.create_provider(&state, source_info(&path)).await?;

        // The geometry column must be annotated with the geoarrow.wkb extension type.
        let field = provider.schema().field_with_name("geometry")?.clone();
        assert_eq!(
            field.metadata().get("ARROW:extension:name").map(String::as_str),
            Some("geoarrow.wkb")
        );

        // The data must be readable end to end.
        let batches = ctx.read_table(provider)?.collect().await?;
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 3);
        Ok(())
    }

    #[tokio::test]
    async fn test_read_plain_parquet_through_geoparquet_format() -> Result<()> {
        let dir = tempfile::tempdir().map_err(|e| DataFusionError::External(e.into()))?;
        let path = dir.path().join("part-0.parquet");
        write_parquet_file(&path, false)?;

        let ctx = SessionContext::new();
        let state = ctx.state();
        let format = GeoParquetTableFormat::default();
        let provider = format.create_provider(&state, source_info(&path)).await?;

        // No geo metadata: the geometry column stays plain binary.
        let field = provider.schema().field_with_name("geometry")?.clone();
        assert!(!field.metadata().contains_key("ARROW:extension:name"));

        let batches = ctx.read_table(provider)?.collect().await?;
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 3);
        Ok(())
    }
}
