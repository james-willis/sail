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
//! `validate`) are also honored: they are extracted from the raw option layers
//! and applied onto [`TableGeoParquetOptions`] (the Parquet resolvers ignore
//! these unknown keys, so they pass through harmlessly).

use std::collections::HashMap;

use datafusion::catalog::Session;
use datafusion::config::ConfigField;
use datafusion_common::{DataFusionError, Result};
use sail_common_datafusion::datasource::OptionLayer;
use sedona_geoparquet::options::TableGeoParquetOptions;

use crate::listing::source::{FormatFactory, ListingTableFormat};
use crate::options::ResolveOptions;
use crate::options::r#gen::{ParquetReadOptions, ParquetWriteOptions};

mod read;
mod write;

pub use read::GeoParquetReadFormat;
pub use write::GeoParquetWriteFormat;

/// GeoParquet-specific option keys, honored in addition to the Parquet options.
pub(super) const GEOPARQUET_OPTION_KEYS: [&str; 4] = [
    "geoparquet_version",
    "overwrite_bbox_columns",
    "geometry_columns",
    "validate",
];

/// Extract the GeoParquet-specific option keys from the raw option layers.
pub(super) fn extract_geoparquet_options(options: &[OptionLayer]) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for layer in options {
        for (key, value) in layer.clone().into_opaque_options() {
            let key = key.to_lowercase();
            if GEOPARQUET_OPTION_KEYS.contains(&key.as_str()) && !value.is_empty() {
                out.insert(key, value);
            }
        }
    }
    out
}

/// Apply extracted GeoParquet-specific options onto a [`TableGeoParquetOptions`].
pub(super) fn apply_geoparquet_options(
    to: &mut TableGeoParquetOptions,
    overrides: &HashMap<String, String>,
) -> Result<()> {
    for (key, value) in overrides {
        to.set(key, value)?;
    }
    Ok(())
}

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
        let geoparquet_options = extract_geoparquet_options(&options);
        let options = ParquetReadOptions::resolve(ctx, options).map_err(DataFusionError::from)?;
        Ok(GeoParquetReadFormat {
            options,
            geoparquet_options,
        })
    }

    fn write(ctx: &dyn Session, options: Vec<OptionLayer>) -> Result<Self::Write> {
        let geoparquet_options = extract_geoparquet_options(&options);
        let options = ParquetWriteOptions::resolve(ctx, options).map_err(DataFusionError::from)?;
        Ok(GeoParquetWriteFormat {
            options,
            geoparquet_options,
        })
    }
}
