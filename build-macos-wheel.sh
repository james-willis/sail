#!/bin/bash
# Build a macOS arm64 (Apple Silicon) wheel for pysail with sedona integration.
# Produces dist/pysail-*-macosx_*_arm64.whl.
#
# Prereqs on the build host:
#   - macOS on Apple Silicon (M1+)
#   - Homebrew with geos installed: `brew install geos`
#   - Rust toolchain (rustup) with aarch64-apple-darwin target
#   - Python 3.11 on PATH (python3.11)
#
# The script bootstraps an isolated venv under .venvs/build and installs
# maturin there — nothing is installed into your global Python.
set -e

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "error: this script must run on macOS arm64" >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "error: Homebrew is required (brew install geos)" >&2
  exit 1
fi

if ! brew list geos >/dev/null 2>&1; then
  echo "error: geos not installed — run: brew install geos" >&2
  exit 1
fi

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "error: python3.11 not found on PATH" >&2
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo/rustup not installed" >&2
  exit 1
fi

# Point pkg-config at brews geos so the build links correctly.
BREW_PREFIX="$(brew --prefix)"
export PKG_CONFIG_PATH="${BREW_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export DYLD_FALLBACK_LIBRARY_PATH="${BREW_PREFIX}/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"

BUILD_VENV=".venvs/build"
if [[ ! -x "${BUILD_VENV}/bin/maturin" || ! -x "${BUILD_VENV}/bin/delocate-wheel" ]]; then
  echo "=== Bootstrapping build venv at ${BUILD_VENV} ==="
  python3.11 -m venv "${BUILD_VENV}"
  "${BUILD_VENV}/bin/pip" install --upgrade pip
  "${BUILD_VENV}/bin/pip" install maturin delocate
fi

mkdir -p dist dist_raw

echo "=== Building wheel (macOS arm64) ==="
export PYO3_PYTHON="$(pwd)/${BUILD_VENV}/bin/python3.11"
"${BUILD_VENV}/bin/maturin" build \
  --release \
  --interpreter "${PYO3_PYTHON}" \
  --out dist_raw

echo "=== Bundling GEOS with delocate ==="
raw_wheel=$(ls -t dist_raw/pysail-*.whl | head -1)
"${BUILD_VENV}/bin/delocate-wheel" --wheel-dir dist -v "${raw_wheel}"
rm -rf dist_raw

echo "=== Done ==="
ls -la dist/*.whl
