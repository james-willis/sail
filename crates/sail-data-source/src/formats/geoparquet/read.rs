use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::datasource::physical_plan::parquet::CachedParquetFileReaderFactory;
use datafusion_common::Result;
use datafusion_common::parsers::CompressionTypeVariant;
use datafusion_common::stats::Precision;
use datafusion_datasource::file_format::FileFormat;
use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use object_store::{ObjectMeta, ObjectStore};
use sail_common_datafusion::schema_evolution::SchemaEvolutionPhysicalExprAdapterFactory;
use sedona_common::option::SedonaOptions;
use sedona_geoparquet::format::{GeoParquetFileSource, GeoParquetFormat};
use sedona_geoparquet::options::TableGeoParquetOptions;

use crate::listing::source::{ListingFileMeta, ListingFileSample, ListingScanInput, ReadFormat};
use crate::options::r#gen::ParquetReadOptions;

#[derive(Debug, Clone)]
pub struct GeoParquetReadFormat {
    pub(super) options: ParquetReadOptions,
    pub(super) geoparquet_options: HashMap<String, String>,
}

impl GeoParquetReadFormat {
    /// Build the underlying SedonaDB [`GeoParquetFormat`] from the resolved
    /// Parquet options. The Parquet options are converted into
    /// [`TableGeoParquetOptions`] (GeoParquet-specific fields keep their
    /// defaults).
    /// Build the effective [`TableGeoParquetOptions`]: the resolved Parquet
    /// options plus the GeoParquet-specific option overrides.
    fn table_geoparquet_options(&self) -> Result<TableGeoParquetOptions> {
        let parquet_options = self.options.clone().into_table_options();
        let mut geo = TableGeoParquetOptions::from(parquet_options);
        super::apply_geoparquet_options(&mut geo, &self.geoparquet_options)?;
        Ok(geo)
    }

    fn geoparquet_format(&self) -> Result<GeoParquetFormat> {
        Ok(GeoParquetFormat::new(self.table_geoparquet_options()?))
    }
}

#[async_trait::async_trait]
impl ReadFormat for GeoParquetReadFormat {
    async fn infer_compression(
        &self,
        _ctx: &dyn Session,
        _files: &[ListingFileSample<'_>],
    ) -> Result<CompressionTypeVariant> {
        Ok(CompressionTypeVariant::UNCOMPRESSED)
    }

    async fn infer_schema(
        &self,
        ctx: &dyn Session,
        files: &[ListingFileSample<'_>],
        _compression: CompressionTypeVariant,
    ) -> Result<SchemaRef> {
        let format = self.geoparquet_format()?;

        // Mirror the Parquet format's per-sample iteration: infer a schema from
        // each sampled group (which may span multiple stores) and merge them.
        // `GeoParquetFormat::infer_schema` merges the `geo` footer metadata
        // across the objects it is given and annotates geometry columns with
        // the `geoarrow.wkb` extension type, so per-group schemas already carry
        // that metadata and merge cleanly.
        let mut schemas: Vec<Schema> = Vec::new();
        for group in files {
            if group.objects.is_empty() {
                continue;
            }
            let schema = format
                .infer_schema(ctx, &group.store, &group.objects)
                .await?;
            schemas.push(schema.as_ref().clone());
        }

        let merged = Schema::try_merge(schemas)?;
        Ok(Arc::new(merged))
    }

    async fn infer_file_meta(
        &self,
        ctx: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        object: &ObjectMeta,
        file_schema: SchemaRef,
        _compression: CompressionTypeVariant,
    ) -> Result<ListingFileMeta> {
        let format = self.geoparquet_format()?;
        let mut statistics = format
            .infer_stats(ctx, store, Arc::clone(&file_schema), object)
            .await?;

        // DataFusion's Parquet opener replaces column references whose file
        // statistics prove them constant (exact min == max, no nulls) with plain
        // literal expressions, and the physical expression simplifier then folds
        // away the metadata-preserving column wrappers that `sedona-geoparquet`
        // uses to keep Arrow extension metadata attached to column references.
        // The resulting literals lose the extension typing, so spatial function
        // kernels no longer match (e.g. `st_astext(binary): No kernel matching
        // arguments` when scanning a file whose geometry column holds a single
        // value). Reporting the min/max and null-count statistics of
        // extension-typed columns as inexact keeps such columns as column
        // references, preserving their extension typing. Min/max statistics on
        // WKB bytes carry no pruning value (spatial pruning uses GeoParquet
        // metadata instead), so this loses nothing.
        for (field, column_statistics) in file_schema
            .fields()
            .iter()
            .zip(statistics.column_statistics.iter_mut())
        {
            if field.metadata().contains_key("ARROW:extension:name") {
                column_statistics.min_value = column_statistics.min_value.clone().to_inexact();
                column_statistics.max_value = column_statistics.max_value.clone().to_inexact();
                column_statistics.null_count = Precision::Absent;
            }
        }

        Ok(ListingFileMeta {
            statistics,
            // GeoParquet's `FileFormat` does not expose file-level orderings the
            // way the Parquet reader does, so none are advertised here.
            ordering: None,
        })
    }

    async fn scan(&self, ctx: &dyn Session, input: ListingScanInput) -> Result<FileScanConfig> {
        // `GeoParquetFormat::create_physical_plan` enriches the `GeoParquetFileSource` with the
        // file-metadata cache, a `CachedParquetFileReaderFactory`, and the `SedonaOptions` bbox
        // bounder used for spatial pruning. Sail's listing planner builds `DataSourceExec`
        // straight from this `FileScanConfig` and never calls `create_physical_plan`, so we
        // reproduce that enrichment here — otherwise geoparquet scans lose footer caching and
        // spatial (bbox) pruning versus the sedona-db native path.
        let mut source = GeoParquetFileSource::new(input.schema, self.table_geoparquet_options()?);
        if let Some(hint) = self
            .options
            .clone()
            .into_table_options()
            .global
            .metadata_size_hint
        {
            source = source.with_metadata_size_hint(hint);
        }
        let cache = ctx.runtime_env().cache_manager.get_file_metadata_cache();
        let store = ctx
            .runtime_env()
            .object_store(input.object_store_url.clone())?;
        source = source
            .with_metadata_cache(Arc::clone(&cache))
            .with_parquet_file_reader_factory(Arc::new(CachedParquetFileReaderFactory::new(
                store, cache,
            )));
        if let Some(sedona_options) = ctx.config_options().extensions.get::<SedonaOptions>() {
            source = source.with_bounder_factory(sedona_options.runtime.bounder_factory().clone());
        }

        let config = FileScanConfigBuilder::new(input.object_store_url, Arc::new(source))
            .with_file_groups(input.file_groups)
            .with_constraints(input.constraints)
            .with_statistics(input.statistics)
            .with_expr_adapter(Some(Arc::new(SchemaEvolutionPhysicalExprAdapterFactory {})))
            .with_projection_indices(input.projection)?
            .with_limit(input.limit)
            .with_output_ordering(input.output_ordering)
            .with_preserve_order(input.preserve_order)
            .with_partitioned_by_file_group(input.partitioned_by_file_group)
            .build();

        Ok(config)
    }

    fn path_glob_filter(&self) -> Option<&str> {
        self.options.path_glob_filter.as_deref()
    }
}
