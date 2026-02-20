#!/bin/bash

# Based on cuda129.sh

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

nvidia-smi

export NVCC="ccache nvcc"

export CUPY_ACCELERATORS="cub,cutensor"

echo "================ Environment Variables ================"
env
echo "======================================================="


trap "$ACTIONS/cleanup.sh" EXIT

# Remove CuPy installed in base Docker image
conda remove --force cupy cupy-core

# Overwrite:
#  - CUPY_NVCC_GENERATE_CODE to accelerate build
#  - CXX to use ccache
CUPY_NVCC_GENERATE_CODE="current" CXX=/usr/lib/ccache/g++ pip install -v ".[test]"
"$ACTIONS/unittest.sh" "not slow" \
    "cupyx_tests/scipy_tests/sparse_tests/csgraph_tests" \
    "cupyx_tests/scipy_tests/spatial_tests/test_kdtree_cuvs.py" \
    "cupyx_tests/scipy_tests/spatial_tests/test_distance.py"
