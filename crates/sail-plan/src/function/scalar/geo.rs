use std::sync::Arc;

use datafusion_expr::ScalarUDF;

use crate::function::common::{ScalarFunction, ScalarFunctionBuilder};

pub(super) fn list_built_in_geo_functions() -> Vec<(&'static str, ScalarFunction)> {
    sail_sedona::sedona_scalar_udfs()
        .map(|(name, udf)| {
            let static_name: &'static str = Box::leak(name.to_string().into_boxed_str());
            let builder: ScalarFunction =
                ScalarFunctionBuilder::scalar_udf(move || -> Arc<ScalarUDF> { udf.clone() });
            (static_name, builder)
        })
        .collect()
}
