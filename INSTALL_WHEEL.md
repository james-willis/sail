# pysail (sedona-integration) — Install Instructions

This is a preview build of `pysail` with SedonaDB geospatial functions integrated.
You'll get a `.whl` file from me; follow the steps below to install and try it.

## Requirements

- **Python 3.11** (the wheel is built specifically for cp311; other Python versions will not install)
- A fresh virtualenv is strongly recommended
- OS-specific wheel:
  - Intel/AMD Linux: `pysail-*-manylinux_2_28_x86_64.whl` (glibc 2.28+, i.e. RHEL 8 / Debian 11 / Ubuntu 20.04 or newer)
  - Apple Silicon Mac (M1+): `pysail-*-macosx_14_0_arm64.whl` (macOS 14 Sonoma or newer)

## CPU compatibility

The Linux wheel is compiled for the **baseline x86_64 ISA** (SSE2 only) — no
AVX/AVX2/AVX-512 or vendor-specific extensions — so it runs on any Intel or
AMD 64-bit CPU made since roughly 2003.

## Linux (Intel/AMD x86_64)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install /path/to/pysail-*-manylinux_2_28_x86_64.whl
```

If `pip install` complains the wheel is "not supported", double-check `python --version` is 3.11.x.

## macOS (Apple Silicon M1+)

The wheel is self-contained — GEOS is bundled inside. No `brew install` needed.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install /path/to/pysail-*-macosx_14_0_arm64.whl
```

> Intel Macs are not supported by this preview wheel. If you're on an Intel Mac, let me know and I'll build a separate wheel.

## Smoke test

```bash
python - <<'PY'
from pysail.spark import SparkConnectServer
from pyspark.sql import SparkSession

server = SparkConnectServer(port=50051)
server.start(background=True)

spark = SparkSession.builder.remote("sc://localhost:50051").getOrCreate()
spark.sql("SELECT ST_Point(1.0, 2.0) AS p").show(truncate=False)
PY
```

You should see a single-row result with a POINT geometry. If that works, the Sedona functions (`ST_*`) are available.

## Troubleshooting

- **`ImportError: libgeos_c.so.1 not found` (Linux)** — the wheel should bundle GEOS via auditwheel. If it didn't, install the system package: `apt-get install libgeos-c1v5` (Debian/Ubuntu) or `dnf install geos` (Fedora/RHEL).
- **Wheel won't install** — confirm `python --version` is 3.11.x; no other 3.x version is supported by this build. On macOS, also confirm you're on macOS 14 (Sonoma) or newer.

## Reporting issues back to me

When something breaks, please send:
- Output of `python --version` and `uname -a`
- The exact `pip install` / runtime error
- The wheel filename you installed
