//! Python bindings for PySail.
//!
//! This module allows Python to interact with the Sail computation engine
//! by binding the Rust functions and types to Python.
mod cli;
mod globals;
mod spark;

use pyo3::prelude::*;

/// Creates the `_native` Python module.
/// Registers the version constant, the `main` function,
/// and various submodules.
/// Configure the PROJ CRS engine for ST_Transform.
///
/// libproj is loaded dynamically at runtime; pass the path to a PROJ shared
/// library (e.g. pyproj's bundled one), its proj.db, and a resource search
/// path. Called automatically from `pysail.spark` when pyproj is installed.
#[pyfunction]
#[pyo3(signature = (shared_library_path=None, database_path=None, search_path=None))]
fn configure_proj_shared(
    shared_library_path: Option<String>,
    database_path: Option<String>,
    search_path: Option<String>,
) -> PyResult<()> {
    sail_sedona::configure_proj_engine(shared_library_path, database_path, search_path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    spark::register_module(m)?;
    m.add_function(wrap_pyfunction!(cli::main, m)?)?;
    m.add_function(wrap_pyfunction!(configure_proj_shared, m)?)?;
    m.add("_SAIL_VERSION", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
