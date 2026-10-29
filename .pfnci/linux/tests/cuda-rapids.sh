#!/bin/bash

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

export CUPY_NVCC_GENERATE_CODE="current"
export NVCC="ccache nvcc"

# rapidsai/base:26.10's conda marks its environment as externally managed
# (PEP 668), so pip refuses to install into it -- including the --user
# install in unittest.sh. This is a throwaway CI container built around a
# single conda env, so allow pip to install into it. Drop this once the
# target stops installing into the base conda env (e.g. uses a venv).
export PIP_BREAK_SYSTEM_PACKAGES=1

# Overwrite:
#  - CXX to use ccache
#  - CUDA_PATH set in conda to build CuPy
CXX=/usr/lib/ccache/g++ CUDA_PATH="/usr/local/cuda" pip install -v ".[test]"
"$ACTIONS/unittest.sh" "not slow" \
    "cupyx_tests/scipy_tests/sparse_tests/csgraph_tests" \
    "cupyx_tests/scipy_tests/spatial_tests/test_kdtree_cuvs.py" \
    "cupyx_tests/scipy_tests/spatial_tests/test_distance.py"
"$ACTIONS/cleanup.sh"
