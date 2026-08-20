# Free-Threaded (No-GIL) Python Support

Sail's Python extension module (`pysail._native`) is declared
free-threading-compatible via `#[pymodule(gil_used = false)]` in
`crates/sail-python/src/lib.rs`. Without that declaration, importing `pysail`
on a free-threaded CPython build (3.13t/3.14t) re-enables the GIL for the
whole process. This document records the audit backing that declaration.

Verified end to end on CPython 3.14t (macOS arm64): `import pysail` keeps
`sys._is_gil_enabled() == False`, and Spark Python UDFs execute concurrently
across partitions inside the in-process Spark Connect server with the GIL off,
with correct results. The regression test lives at
`python/pysail/tests/spark/test_free_threading.py`.

## Why an audit is needed

On a GIL build, `Python::attach` (pyo3) grants mutual exclusion: at most one
thread runs Python code at a time, so code can (accidentally) rely on the GIL
as a mutex around shared state. On a free-threaded build, `Python::attach`
only attaches the thread to the interpreter — many threads run Python
concurrently. Anything that used the GIL as a lock is a data race.

The crates that touch Python were audited: `sail-python`, `sail-python-udf`,
and the Python data source support in `sail-data-source`
(`src/formats/python`), including all ~40 `Python::attach` /
`attach_persistent` call sites and the embedded Python helper modules
(`spark.py`, `discovery.py`).

## Sound by construction

- **Module caches** — `PySpark::module`
  (`sail-python-udf/src/python/spark.rs`) and `PyDiscovery::module`
  (`sail-data-source/src/formats/python/discovery.rs`) cache the embedded
  Python module in a `pyo3::sync::PyOnceLock`, pyo3 0.26's free-threading-safe
  once-cell (backed by `once_cell::sync::OnceCell`; it detaches from the
  interpreter before blocking, so it cannot deadlock and initializes exactly
  once).
- **Per-UDF Python object caches** — `LazyPyObject`
  (`sail-python-udf/src/lazy.rs`) wraps `PyOnceLock<Py<PyAny>>`. Used by
  `PySparkUDF`, `PySparkGroupAggregateUDF`, `PySparkGroupMapUDF`, and
  `PySparkCoGroupMapUDF` to lazily build the Python wrapper object that all
  worker threads then share.
- **Global state** — `GLOBALS: PyOnceLock<GlobalState>`
  (`sail-python/src/globals.rs`) for the runtime/telemetry singleton.
- **Data source registry** — `DATA_SOURCE_REGISTRY` (`discovery.rs`) is a
  `once_cell::Lazy<DashMap>`; registration uses the `DashMap` entry API for
  atomic check-and-insert (no TOCTOU).
- **`PythonDataSource`** (`datasource.rs`) — schema cache is a
  `once_cell::sync::OnceCell`; the deserialized datasource cache is a
  `Mutex<Option<Py<PyAny>>>`. Real locks, no GIL reliance.
- **Streams** — `PyMapStream`/`PyInputStream` (`sail-python-udf/src/stream.rs`)
  and the data source executor streams use tokio channels and mutexes;
  `#[pyclass]` types rely on pyo3's runtime borrow checking, which is
  atomic-based and free-threading-safe in pyo3 0.26.
- **Accumulators** — `BatchAggregateAccumulator` mutates only through
  `&mut self`; DataFusion guarantees exclusive per-accumulator access.
- **Attach sites** — every `Python::attach` / `attach_persistent` closure in
  the audited crates operates on per-call locals (arguments in, converted
  values out) or on the lock-based caches above. None mutate shared state
  under the assumption that attaching excludes other threads.
- **Embedded Python helpers** — the wrapper classes in
  `sail-python-udf/src/python/spark.py` (e.g. `PySparkBatchUdf`) set all
  instance attributes in `__init__` and never mutate `self` in `__call__`, so
  a single shared wrapper instance may be called from many threads. Module
  level names are import-time constants. The PySpark serializer objects held
  by the wrappers are only used through read-only conversion calls.

## `attach_persistent` under free-threading

`attach_persistent` (`sail-python-udf/src/threadstate.rs`) pins each worker
thread's `PyThreadState` for the lifetime of the OS thread, fixing a
use-after-free with libraries that key native resources by thread (pyproj's
per-thread `PJ_CONTEXT*`; see lakehq/sail#2456, upstream PR #2457).

Conclusion of the re-validation: **the mechanism is still necessary and still
correct on free-threaded builds, unchanged.** The `PyGILState_*` API keeps the
same thread-state *lifecycle* semantics on free-threaded CPython — `Ensure`
binds/creates the per-thread state and increments its gilstate counter, and
the outermost `Release` still destroys the auto thread state. Free-threading
removes only the mutual exclusion. Without the pin, a runtime worker thread
would still get a fresh, short-lived `PyThreadState` per UDF call, its
`threading.local` values would still be finalized per call, and the same TSS
dangling-pointer class would apply. `PyEval_SaveThread` (used to detach after
pinning) detaches only the current thread without blocking others, so the pin
is equally cheap with the GIL disabled.

## Building and testing

- Wheels: `.github/workflows/build-wheels.yml` has `*-freethreaded` jobs that
  build cp314t wheels with `maturin ... -i python3.14t`. pyo3 automatically
  drops the `abi3-py38` feature for a free-threaded target, so the result is
  a versioned (non-abi3) wheel. Locally:
  `maturin build --release -i python3.14t`.
- Test: `PYTHON_GIL=0 python3.14t -m pytest python/pysail/tests/spark/test_free_threading.py`
  (in a 3.14t venv with the cp314t wheel, `pyspark-client`, `pandas`,
  `pyarrow` installed). The test skips itself on GIL builds or when the GIL
  is enabled.

## Known boundaries (out of scope here)

- **grpcio is not free-threading-safe.** The Spark Connect *client*
  (`pyspark-client`) imports `grpcio`, which does not declare free-threading
  support, so importing it re-enables the GIL unless the process is started
  with `PYTHON_GIL=0` (upstream: grpc/grpc#38762). Server-side UDF execution
  does not touch grpcio's Python bindings, but any in-process client/server
  test setup needs `PYTHON_GIL=0`. This is upstream-blocked.
- **User workload native-library thread-safety.** With the GIL off, UDF code
  and the native libraries it uses (numpy, shapely/GEOS, pyproj, pandas, ...)
  genuinely run concurrently. Whether a given UDF's dependencies are
  thread-safe is a property of the workload, not of Sail; Sail already ran
  UDFs from multiple threads on GIL builds (the GIL is released around many
  native calls), free-threading just widens the exposure.
- **Deployment runtimes** that ship a GIL CPython (e.g. Python 3.12) simply
  keep using the abi3 wheel; the free-threaded wheel is additive.
