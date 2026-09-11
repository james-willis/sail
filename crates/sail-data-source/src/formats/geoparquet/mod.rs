//! GeoParquet table format backed by the `sedona-geoparquet` crate.
//!
//! This adapts SedonaDB's [`GeoParquetFormat`](sedona_geoparquet::format::GeoParquetFormat)
//! (a DataFusion `FileFormat` wrapping the built-in Parquet format) into Sail's
//! generic [`ListingTableFormat`] / [`FormatFactory`] machinery, so that Spark
//! Connect clients can use `spark.read.format("geoparquet")` and
//! `df.write.format("geoparquet")`.
//!
//! On read, the format inspects the `geo` key/value metadata in Parquet file
//! footers and annotates matching columns with the `geoarrow.wkb` Arrow
//! extension type, which SedonaDB's spatial functions and the spatial join
//! planner understand. Plain Parquet files (no `geo` metadata) read fine and
//! behave exactly like the regular Parquet format.
//!
//! All regular Parquet options are supported and resolved through Sail's
//! Parquet option resolution ([`ParquetReadOptions`] / [`ParquetWriteOptions`]),
//! then converted into [`TableGeoParquetOptions`]. GeoParquet-specific option
//! keys (`geoparquet_version`, `overwrite_bbox_columns`, `geometry_columns`,
//! `validate`) are not currently plumbed through and fall back to their
//! defaults (GeoParquet 1.0 on write); the underlying option resolvers ignore
//! unknown keys rather than failing.

use datafusion::catalog::Session;
use datafusion_common::{DataFusionError, Result};
use sail_common_datafusion::datasource::OptionLayer;

use crate::listing::source::{FormatFactory, ListingTableFormat};
use crate::options::ResolveOptions;
use crate::options::r#gen::{ParquetReadOptions, ParquetWriteOptions};

mod read;
mod write;

pub use read::GeoParquetReadFormat;
pub use write::GeoParquetWriteFormat;

pub type GeoParquetTableFormat = ListingTableFormat<GeoParquetFormatFactory>;

#[derive(Debug, Default)]
pub struct GeoParquetFormatFactory;

impl FormatFactory for GeoParquetFormatFactory {
    type Read = GeoParquetReadFormat;
    type Write = GeoParquetWriteFormat;

    fn name() -> &'static str {
        "geoparquet"
    }

    fn read(ctx: &dyn Session, options: Vec<OptionLayer>) -> Result<Self::Read> {
        let options = ParquetReadOptions::resolve(ctx, options).map_err(DataFusionError::from)?;
        Ok(GeoParquetReadFormat { options })
    }

    fn write(ctx: &dyn Session, options: Vec<OptionLayer>) -> Result<Self::Write> {
        let options = ParquetWriteOptions::resolve(ctx, options).map_err(DataFusionError::from)?;
        Ok(GeoParquetWriteFormat { options })
    }
}
