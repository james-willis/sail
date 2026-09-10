use std::sync::Arc;

use datafusion_expr::ScalarUDF;

use crate::function::common::{ScalarFunction, ScalarFunctionBuilder};

/// All scalar geospatial functions come from sedona-db (the `sail-sedona` crate),
/// wrapped as Sail scalar functions.
pub(super) fn list_built_in_geo_functions() -> Vec<(&'static str, ScalarFunction)> {
    sail_sedona::sedona_scalar_udfs()
        .map(|(name, udf)| {
            let builder: ScalarFunction =
                ScalarFunctionBuilder::scalar_udf(move || -> Arc<ScalarUDF> { udf.clone() });
            (name, builder)
        })
        .collect()
}
