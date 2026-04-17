#!/bin/bash
# Build a manylinux aarch64 wheel for pysail with sedona integration.
set -e

docker run --rm \
  --platform linux/arm64 \
  -v "$(pwd)":/io \
  -w /io \
  quay.io/pypa/manylinux_2_28_aarch64:latest \
  bash -c '
    set -e
    echo "=== Installing system deps ==="
    dnf install -y geos-devel cmake openssl-devel perl-IPC-Cmd 2>/dev/null || \
      yum install -y geos-devel cmake3 openssl-devel 2>/dev/null || true

    # If geos-devel not in repo, build from source
    if ! pkg-config --exists geos 2>/dev/null; then
      echo "Building GEOS from source..."
      curl -L https://download.osgeo.org/geos/geos-3.13.1.tar.bz2 | tar xj
      cd geos-3.13.1
      mkdir build && cd build
      cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release
      make -j$(nproc)
      make install
      cd /io
      export PKG_CONFIG_PATH=/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
      export LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib:$LD_LIBRARY_PATH
    fi

    echo "=== Installing Rust ==="
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source $HOME/.cargo/env

    echo "=== Installing maturin ==="
    /opt/python/cp311-cp311/bin/pip install maturin

    echo "=== Building wheel ==="
    export PYO3_PYTHON=/opt/python/cp311-cp311/bin/python3.11
    /opt/python/cp311-cp311/bin/maturin build \
      --release \
      --interpreter /opt/python/cp311-cp311/bin/python3.11 \
      --out /io/dist/

    echo "=== Done ==="
    ls -la /io/dist/*.whl
  '
