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

/// The module is declared free-threading-compatible (`gil_used = false`).
/// Without this declaration, importing `pysail` on a free-threaded (no-GIL)
/// CPython build would re-enable the GIL for the whole process. The assertion
/// is backed by an audit of the Python-bridge crates (`sail-python`,
/// `sail-python-udf`, and the Python data source support in
/// `sail-data-source`): all cached Python state uses `PyOnceLock` or
/// lock-based data structures instead of relying on the GIL for mutual
/// exclusion. See `FREE_THREADING.md` at the repository root.
#[pymodule(gil_used = false)]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    spark::register_module(m)?;
    m.add_function(wrap_pyfunction!(cli::main, m)?)?;
    m.add_function(wrap_pyfunction!(configure_proj_shared, m)?)?;
    m.add("_SAIL_VERSION", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
