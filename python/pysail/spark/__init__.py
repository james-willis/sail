from __future__ import annotations

from pysail import _native

__all__ = [
    "SparkConnectServer",
]

_PROJ_CONFIGURED = False


def _configure_proj_from_pyproj() -> None:
    """Best-effort: point ST_Transform's CRS engine at pyproj's bundled PROJ.

    libproj is loaded dynamically at runtime. If pyproj is installed, its
    wheel ships both the shared library and proj.db, so ST_Transform works
    without any system PROJ. Silently does nothing if pyproj is missing;
    users can call ``pysail._native.configure_proj_shared`` manually to use
    a system PROJ instead.
    """
    global _PROJ_CONFIGURED  # noqa: PLW0603
    if _PROJ_CONFIGURED:
        return
    try:
        import sys
        from pathlib import Path

        import pyproj

        data_dir = Path(pyproj.datadir.get_data_dir())
        candidates = []
        if sys.platform == "darwin":
            dylibs_dir = Path(pyproj.__file__).parent / ".dylibs"
            if dylibs_dir.exists():
                candidates.extend(dylibs_dir.glob("libproj*.dylib*"))
        else:
            libs_dir = Path(pyproj.__file__).parent.parent / "pyproj.libs"
            if libs_dir.exists():
                candidates.extend(libs_dir.glob("libproj*.so*"))
        if candidates:
            _native.configure_proj_shared(
                str(candidates[0]),
                str(data_dir / "proj.db"),
                str(data_dir),
            )
            _PROJ_CONFIGURED = True
    except Exception:  # noqa: BLE001, S110
        pass


class SparkConnectServer:
    """The Spark Connect server that uses Sail as the computation engine."""

    def __init__(self, ip: str = "127.0.0.1", port: int = 0) -> None:
        """Create a new Spark Connect server.
        By default, the server will bind to localhost on a random port.

        :param ip: The IP address to bind the server to.
        :param port: The port to bind the server to.
        """
        _configure_proj_from_pyproj()
        self._inner = _native.spark.SparkConnectServer(ip, port)

    def start(self, *, background=True) -> None:
        """Start the server.

        :param background: Whether to start the server in a background thread.
        """
        self._inner.start(background=background)

    def stop(self) -> None:
        """Stop the server."""
        self._inner.stop()

    @property
    def listening_address(self) -> tuple[str, int] | None:
        """The address that the server is listening on,
        or ``None`` if the server is not running.
        The address is a tuple of the IP address and port.
        """
        return self._inner.listening_address

    @property
    def running(self) -> bool:
        """Whether the server is running."""
        return self._inner.running
