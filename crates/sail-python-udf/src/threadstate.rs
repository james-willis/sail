use std::cell::Cell;

use pyo3::{ffi, Python};

thread_local! {
    static THREAD_STATE_PINNED: Cell<bool> = const { Cell::new(false) };
}

/// Keep the current OS thread's Python thread state alive for the lifetime of
/// the thread, then run `f` attached to the interpreter.
///
/// Background: [`Python::attach`] uses `PyGILState_Ensure`/`PyGILState_Release`
/// under the hood. On a thread that is not otherwise attached, the outermost
/// `PyGILState_Release` **destroys** the auto thread state that `Ensure`
/// created. Runtime worker threads therefore observe a fresh, short-lived
/// `PyThreadState` for every UDF invocation.
///
/// That breaks Python libraries that key native resources by thread: values
/// stored in `threading.local` live on the thread state and are finalized when
/// it is destroyed, while companion pointers stored in OS-level TSS
/// (`PyThread_tss_*`) survive with the OS thread. A library that frees a
/// native handle from a `threading.local` finalizer but caches the raw pointer
/// in TSS (pyproj's per-thread `PJ_CONTEXT*` is one example) is left with a
/// dangling pointer that the *next* UDF call on the same worker thread will
/// dereference — a use-after-free that surfaces as heap corruption far from
/// the cause.
///
/// The fix: before the first attach on each worker thread, register the thread
/// state once via `PyGILState_Ensure` and never release that reference, so the
/// thread state (and everything keyed on it) lives exactly as long as the OS
/// thread. This mirrors what a plain Python `threading.Thread` provides and
/// what Python libraries assume. The cost is one retained thread state per
/// pool thread.
///
/// Free-threading (no-GIL) note: this reasoning is unchanged on free-threaded
/// CPython builds. The `PyGILState_*` API keeps the same thread-state
/// lifecycle semantics there — `Ensure` binds (and creates if needed) the
/// per-thread `PyThreadState` and increments its gilstate counter, and the
/// outermost `Release` still destroys the auto thread state; free-threading
/// removes only the mutual exclusion, not the lifecycle. Without the pin,
/// worker threads on a free-threaded interpreter would still observe a fresh
/// short-lived thread state per UDF call, and the same `threading.local` /
/// TSS use-after-free class would apply. `PyEval_SaveThread` detaches the
/// current thread state without blocking other attached threads, so the pin
/// is equally cheap and correct with the GIL disabled.
pub(crate) fn attach_persistent<F, R>(f: F) -> R
where
    F: for<'py> FnOnce(Python<'py>) -> R,
{
    pin_current_thread();
    Python::attach(f)
}

fn pin_current_thread() {
    THREAD_STATE_PINNED.with(|pinned| {
        if pinned.get() {
            return;
        }
        // SAFETY: the interpreter is guaranteed to be initialized here (this
        // code runs while executing a query inside an embedded interpreter or
        // one started by the host process), and we are not in interpreter
        // finalization (query execution has the server alive).
        unsafe {
            let state = ffi::PyGILState_Ensure();
            if matches!(state, ffi::PyGILState_STATE::PyGILState_UNLOCKED) {
                // We acquired the GIL and `Ensure` bound a thread state to
                // this thread (creating it if needed), with the gilstate
                // recursion counter now at least 1. Intentionally skip the
                // matching `PyGILState_Release`: detach and release the GIL
                // while leaving the thread state registered in TSS. Later
                // `PyGILState_Ensure` calls find and reuse it, and their
                // matching releases can never drop the counter to zero, so
                // the thread state is never destroyed.
                let tstate = ffi::PyEval_SaveThread();
                let _ = tstate;
            }
            // If the GIL was already held (`PyGILState_LOCKED`), an outer
            // scope owns the thread state; the unreleased `Ensure` reference
            // taken above still pins it for this thread's lifetime, and we
            // must not touch the GIL that the caller is holding.
        }
        pinned.set(true);
    });
}
