#!/bin/bash

set -uex

# Fetch the CuPy wheel built by GHA (.github/workflows/build-wheel.yml) for the
# current PR HEAD / merge commit, then pip-install it. The wheel-build matrix
# does not yet cover every FlexCI (CUDA, Python) tuple; this script will fail
# on combinations without a matching artifact -- by design, until the matrix
# is expanded.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."; pwd)"

if [[ -z "${GH_TOKEN:-}" ]]; then
    echo "Error: GH_TOKEN env var is required to fetch wheel artifacts from cupy/cupy CI" >&2
    exit 1
fi

SHA=$(git -C "${REPO_ROOT}" rev-parse HEAD)
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
CUDA_VER=$(python3 -c "import json; print(json.load(open('/usr/local/cuda/version.json'))['cuda']['version'])")
CUDA_MAJOR="${CUDA_VER%%.*}"
ARTIFACT_NAME="cupy-cuda${CUDA_MAJOR}x-py${PY_VER}-linux-64-${SHA}"

RUN_ID=$("${REPO_ROOT}/ci/tools/lookup-run-id" --commit "${SHA}" cupy/cupy "CI")

WHEEL_DIR="$(mktemp -d)"
trap 'rm -rf "${WHEEL_DIR}"' EXIT
gh run download "${RUN_ID}" -R cupy/cupy -n "${ARTIFACT_NAME}" -D "${WHEEL_DIR}"

WHEEL="$(ls "${WHEEL_DIR}"/*.whl | head -n 1)"
time python3 -m pip install --user -v "${WHEEL}[test]"
