#!/bin/bash

# Based on cuda122.sh

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

export NVCC="ccache nvcc"

export CUPY_ACCELERATORS="cutensor,cub"

# Make the build faster.
export CUPY_NVCC_GENERATE_CODE="arch=compute_70,code=sm_70"

"$ACTIONS/build.sh"

# Make sure that CuPy can be imported without CUDA Toolkit installed.
rm -rf /usr/local/cuda*
pushd tests/import_tests
python3 test_import.py
popd

"$ACTIONS/cleanup.sh"
