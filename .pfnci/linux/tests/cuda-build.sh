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

# Remove system-wide CUDA Toolkit from container.
rm -rf /usr/local/cuda*

# Make sure that CuPy can be imported without CUDA Toolkit installed.
pushd tests/import_tests
python3 test_import.py
popd

# Install CUDA Toolkit using Pip Wheels, and confirm CuPy works.
# TODO(kmaehashi): Install with [ctk] extra.
# TODO(kmaehashi): Consider better tests.
python3 -m pip install --user 'cuda-toolkit[cudart,nvrtc,cublas,cufft,cusolver,cusparse,curand]==12.*'
pushd tests
python3 -c 'import cupy; ok = cupy.arange(10).sum().item() == 45; print(ok); assert ok'
popd

"$ACTIONS/cleanup.sh"
