#!/bin/bash

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

export NVCC="ccache nvcc"

export CUPY_USE_CUDA_PYTHON="1"

python3 -m pip install --user Cython
python3 -m pip install --user https://github.com/NVIDIA/cuda-python/archive/refs/heads/main.tar.gz

"$ACTIONS/build.sh"
"$ACTIONS/unittest.sh" "not slow"
"$ACTIONS/cleanup.sh"
