#!/bin/bash

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

export CUPY_NVCC_GENERATE_CODE="current"
export NVCC="ccache nvcc"

# Overwrite:
#  - CXX to use ccache
#  - CUDA_PATH set in conda to build CuPy
CXX=/usr/lib/ccache/g++ CUDA_PATH="/usr/local/cuda" pip install -v ".[test]"
"$ACTIONS/unittest.sh" "not slow" \
    "cupyx_tests/scipy_tests/sparse_tests/csgraph_tests" \
    "cupyx_tests/scipy_tests/spatial_tests"
"$ACTIONS/cleanup.sh"
