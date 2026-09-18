use std::sync::Arc;

use sail_common_datafusion::catalog::{DatabaseStatus, TableStatus};
use tokio::runtime::Handle;

use super::{
    AlterTableOptions, CatalogProvider, CreateDatabaseOptions, CreateTableMetadataRequirement,
    CreateTableOptions, CreateViewOptions, DropDatabaseOptions, DropTableOptions, DropViewOptions,
    Namespace,
};
use crate::error::{CatalogError, CatalogResult};
use crate::lakehouse::{
    BeginTableAccessRequest, DeltaRatifiedCommitRequest, DeltaRatifiedCommitResponse,
    LakehouseCapability, LakehouseCommitOutcome, LakehouseCommitRequest, LakehouseCreatePlan,
    LakehouseCreateRequest, LakehouseResolvedTable, LakehouseScanPlanningRequest,
    LakehouseScanPlanningResponse, ResolveLakehouseTableRequest, TableAccessSession,
};

pub struct RuntimeAwareCatalogProvider<P: CatalogProvider> {
    inner: Arc<P>,
    handle: Handle,
}

impl<P: CatalogProvider> RuntimeAwareCatalogProvider<P> {
    pub fn try_new<F>(initializer: F, handle: Handle) -> CatalogResult<Self>
    where
        F: FnOnce() -> CatalogResult<P>,
    {
        let _guard = handle.enter();
        let inner = Arc::new(initializer()?);
        Ok(Self { inner, handle })
    }
}

#[async_trait::async_trait]
impl<P: CatalogProvider + 'static> CatalogProvider for RuntimeAwareCatalogProvider<P> {
    fn derives_table_location(&self) -> bool {
        // Forward: the wrapped provider decides whether the catalog assigns table
        // locations itself (an Iceberg REST catalog does). Without this the wrapper
        // silently answered the trait default and CREATE TABLE fell back to the
        // local warehouse directory.
        self.inner.derives_table_location()
    }

    fn get_name(&self) -> &str {
        self.inner.get_name()
    }

    async fn create_database(
        &self,
        database: &Namespace,
        options: CreateDatabaseOptions,
    ) -> CatalogResult<DatabaseStatus> {
        let inner = self.inner.clone();
        let database = database.clone();
        self.handle
            .spawn(async move { inner.create_database(&database, options).await })
            .await
            .map_err(|e| {
                CatalogError::External(format!("Failed to execute create_database: {e}"))
            })?
    }

    async fn get_database(&self, database: &Namespace) -> CatalogResult<DatabaseStatus> {
        let inner = self.inner.clone();
        let database = database.clone();
        self.handle
            .spawn(async move { inner.get_database(&database).await })
            .await
            .map_err(|e| CatalogError::External(format!("Failed to execute get_database: {e}")))?
    }

    async fn list_databases(
        &self,
        prefix: Option<&Namespace>,
    ) -> CatalogResult<Vec<DatabaseStatus>> {
        let inner = self.inner.clone();
        let prefix = prefix.cloned();
        self.handle
            .spawn(async move { inner.list_databases(prefix.as_ref()).await })
            .await
            .map_err(|e| CatalogError::External(format!("Failed to execute list_databases: {e}")))?
    }

    async fn drop_database(
        &self,
        database: &Namespace,
        options: DropDatabaseOptions,
    ) -> CatalogResult<()> {
        let inner = self.inner.clone();
        let database = database.clone();
        self.handle
            .spawn(async move { inner.drop_database(&database, options).await })
            .await
            .map_err(|e| CatalogError::External(format!("Failed to execute drop_database: {e}")))?
    }

    async fn create_table(
        &self,
        database: &Namespace,
        table: &str,
        options: CreateTableOptions,
    ) -> CatalogResult<TableStatus> {
        let inner = self.inner.clone();
        let database = database.clone();
        let table = table.to_string();
        self.handle
            .spawn(async move { inner.create_table(&database, &table, options).await })
            .await
            .map_err(|e| CatalogError::External(format!("Failed to execute create_table: {e}")))?
    }

    fn create_table_metadata_requirement(
        &self,
        options: &CreateTableOptions,
    ) -> CatalogResult<CreateTableMetadataRequirement> {
        self.inner.create_table_metadata_requirement(options)
    }

    fn lakehouse_capabilities(&self) -> Vec<LakehouseCapability> {
        self.inner.lakehouse_capabilities()
    }

    async fn resolve_lakehouse_table(
        &self,
        database: &Namespace,
        table: &str,
        request: ResolveLakehouseTableRequest,
    ) -> CatalogResult<LakehouseResolvedTable> {
        let inner = self.inner.clone();
        let database = database.clone();
        let table = table.to_string();
        self.handle
            .spawn(async move {
                inner
                    .resolve_lakehouse_table(&database, &table, request)
                    .await
            })
            .await
            .map_err(|e| {
                CatalogError::External(format!("Failed to execute resolve_lakehouse_table: {e}"))
            })?
    }

    async fn plan_lakehouse_create(
        &self,
        database: &Namespace,
        table: &str,
        request: LakehouseCreateRequest,
    ) -> CatalogResult<LakehouseCreatePlan> {
        let inner = self.inner.clone();
        let database = database.clone();
        let table = table.to_string();
        self.handle
            .spawn(async move {
                inner
                    .plan_lakehouse_create(&database, &table, request)
                    .await
            })
            .await
            .map_err(|e| {
                CatalogError::External(format!("Failed to execute plan_lakehouse_create: {e}"))
            })?
    }

    async fn begin_table_access(
        &self,
        database: &Namespace,
        table: &str,
        request: BeginTableAccessRequest,
    ) -> CatalogResult<TableAccessSession> {
        let inner = self.inner.clone();
        let database = database.clone();
        let table = table.to_string();
        self.handle
            .spawn(async move { inner.begin_table_access(&database, &table, request).await })
            .await
            .map_err(|e| {
                CatalogError::External(format!("Failed to execute begin_table_access: {e}"))
            })?
    }

    async fn plan_lakehouse_scan(
        &self,
        database: &Namespace,
        table: &str,
        request: LakehouseScanPlanningRequest,
    ) -> CatalogResult<LakehouseScanPlanningResponse> {
        let inner = self.inner.clone();
        let database = database.clone();
        let table = table.to_string();
        self.handle
            .spawn(async move { inner.plan_lakehouse_scan(&database, &table, request).await })
            .await
            .map_err(|e| {
                CatalogError::External(format!("Failed to execute plan_lakehouse_scan: {e}"))
            })?
    }

    async fn commit_lakehouse_table(
        &self,
        database: &Namespace,
        table: &str,
        request: LakehouseCommitRequest,
    ) -> CatalogResult<LakehouseCommitOutcome> {
        let inner = self.inner.clone();
        let database = database.clone();
        let table = table.to_string();
        self.handle
            .spawn(async move {
                inner
                    .commit_lakehouse_table(&database, &table, request)
                    .await
            })
            .await
            .map_err(|e| {
                CatalogError::External(format!("Failed to execute commit_lakehouse_table: {e}"))
            })?
    }

    async fn get_delta_ratified_commits(
        &self,
        database: &Namespace,
        table: &str,
        request: DeltaRatifiedCommitRequest,
    ) -> CatalogResult<DeltaRatifiedCommitResponse> {
        let inner = self.inner.clone();
        let database = database.clone();
        let table = table.to_string();
        self.handle
            .spawn(async move {
                inner
                    .get_delta_ratified_commits(&database, &table, request)
                    .await
            })
            .await
            .map_err(|e| {
                CatalogError::External(format!("Failed to execute get_delta_ratified_commits: {e}"))
            })?
    }

    async fn get_table(&self, database: &Namespace, table: &str) -> CatalogResult<TableStatus> {
        let inner = self.inner.clone();
        let database = database.clone();
        let table = table.to_string();
        self.handle
            .spawn(async move { inner.get_table(&database, &table).await })
            .await
            .map_err(|e| CatalogError::External(format!("Failed to execute get_table: {e}")))?
    }

    async fn list_tables(&self, database: &Namespace) -> CatalogResult<Vec<TableStatus>> {
        let inner = self.inner.clone();
        let database = database.clone();
        self.handle
            .spawn(async move { inner.list_tables(&database).await })
            .await
            .map_err(|e| CatalogError::External(format!("Failed to execute list_tables: {e}")))?
    }

    async fn drop_table(
        &self,
        database: &Namespace,
        table: &str,
        options: DropTableOptions,
    ) -> CatalogResult<()> {
        let inner = self.inner.clone();
        let database = database.clone();
        let table = table.to_string();
        self.handle
            .spawn(async move { inner.drop_table(&database, &table, options).await })
            .await
            .map_err(|e| CatalogError::External(format!("Failed to execute drop_table: {e}")))?
    }

    async fn alter_table(
        &self,
        database: &Namespace,
        table: &str,
        options: AlterTableOptions,
    ) -> CatalogResult<()> {
        let inner = self.inner.clone();
        let database = database.clone();
        let table = table.to_string();
        self.handle
            .spawn(async move { inner.alter_table(&database, &table, options).await })
            .await
            .map_err(|e| CatalogError::External(format!("Failed to execute alter_table: {e}")))?
    }

    async fn create_view(
        &self,
        database: &Namespace,
        view: &str,
        options: CreateViewOptions,
    ) -> CatalogResult<TableStatus> {
        let inner = self.inner.clone();
        let database = database.clone();
        let view = view.to_string();
        self.handle
            .spawn(async move { inner.create_view(&database, &view, options).await })
            .await
            .map_err(|e| CatalogError::External(format!("Failed to execute create_view: {e}")))?
    }

    async fn get_view(&self, database: &Namespace, view: &str) -> CatalogResult<TableStatus> {
        let inner = self.inner.clone();
        let database = database.clone();
        let view = view.to_string();
        self.handle
            .spawn(async move { inner.get_view(&database, &view).await })
            .await
            .map_err(|e| CatalogError::External(format!("Failed to execute get_view: {e}")))?
    }

    async fn list_views(&self, database: &Namespace) -> CatalogResult<Vec<TableStatus>> {
        let inner = self.inner.clone();
        let database = database.clone();
        self.handle
            .spawn(async move { inner.list_views(&database).await })
            .await
            .map_err(|e| CatalogError::External(format!("Failed to execute list_views: {e}")))?
    }

    async fn drop_view(
        &self,
        database: &Namespace,
        view: &str,
        options: DropViewOptions,
    ) -> CatalogResult<()> {
        let inner = self.inner.clone();
        let database = database.clone();
        let view = view.to_string();
        self.handle
            .spawn(async move { inner.drop_view(&database, &view, options).await })
            .await
            .map_err(|e| CatalogError::External(format!("Failed to execute drop_view: {e}")))?
    }
}

#[cfg(test)]
mod derives_table_location_tests {
    #![allow(unused_variables)]
    use super::*;
    use crate::provider::*;

    /// A provider that says it derives table locations itself, wrapped the way the
    /// session wraps every provider. The wrapper must forward the answer: when it fell
    /// back to the trait default, `CREATE TABLE` in an Iceberg REST catalog invented a
    /// path under the local warehouse directory instead of letting the catalog choose.
    struct DerivingProvider;

    #[async_trait::async_trait]
    impl CatalogProvider for DerivingProvider {
        fn get_name(&self) -> &str {
            "deriving"
        }

        fn derives_table_location(&self) -> bool {
            true
        }

        async fn create_database(
            &self,
            database: &Namespace,
            options: CreateDatabaseOptions,
        ) -> CatalogResult<DatabaseStatus> {
            unimplemented!("create_database is not exercised by this test")
        }

        async fn get_database(&self, database: &Namespace) -> CatalogResult<DatabaseStatus> {
            unimplemented!("get_database is not exercised by this test")
        }

        async fn list_databases(
            &self,
            prefix: Option<&Namespace>,
        ) -> CatalogResult<Vec<DatabaseStatus>> {
            unimplemented!("list_databases is not exercised by this test")
        }

        async fn drop_database(
            &self,
            database: &Namespace,
            options: DropDatabaseOptions,
        ) -> CatalogResult<()> {
            unimplemented!("drop_database is not exercised by this test")
        }

        async fn create_table(
            &self,
            database: &Namespace,
            table: &str,
            options: CreateTableOptions,
        ) -> CatalogResult<TableStatus> {
            unimplemented!("create_table is not exercised by this test")
        }

        async fn get_table(&self, database: &Namespace, table: &str) -> CatalogResult<TableStatus> {
            unimplemented!("get_table is not exercised by this test")
        }

        async fn list_tables(&self, database: &Namespace) -> CatalogResult<Vec<TableStatus>> {
            unimplemented!("list_tables is not exercised by this test")
        }

        async fn drop_table(
            &self,
            database: &Namespace,
            table: &str,
            options: DropTableOptions,
        ) -> CatalogResult<()> {
            unimplemented!("drop_table is not exercised by this test")
        }

        async fn alter_table(
            &self,
            database: &Namespace,
            table: &str,
            options: AlterTableOptions,
        ) -> CatalogResult<()> {
            unimplemented!("alter_table is not exercised by this test")
        }

        async fn create_view(
            &self,
            database: &Namespace,
            view: &str,
            options: CreateViewOptions,
        ) -> CatalogResult<TableStatus> {
            unimplemented!("create_view is not exercised by this test")
        }

        async fn get_view(&self, database: &Namespace, view: &str) -> CatalogResult<TableStatus> {
            unimplemented!("get_view is not exercised by this test")
        }

        async fn list_views(&self, database: &Namespace) -> CatalogResult<Vec<TableStatus>> {
            unimplemented!("list_views is not exercised by this test")
        }

        async fn drop_view(
            &self,
            database: &Namespace,
            view: &str,
            options: DropViewOptions,
        ) -> CatalogResult<()> {
            unimplemented!("drop_view is not exercised by this test")
        }
    }

    #[tokio::test]
    async fn runtime_aware_wrapper_forwards_derives_table_location() {
        let wrapped = RuntimeAwareCatalogProvider::try_new(
            || Ok(DerivingProvider),
            tokio::runtime::Handle::current(),
        )
        .expect("wrap");
        assert!(wrapped.derives_table_location());
    }
}
