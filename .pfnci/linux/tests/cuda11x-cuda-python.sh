#!/bin/bash

# AUTO GENERATED: DO NOT EDIT!

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

nvidia-smi

export NVCC="ccache nvcc"

export CUPY_ACCELERATORS="cutensor,cub"

export CUPY_USE_CUDA_PYTHON="1"

echo "================ Environment Variables ================"
env
echo "======================================================="

"$ACTIONS/build.sh"
"$ACTIONS/unittest.sh" "not slow and not multi_gpu"
"$ACTIONS/cleanup.sh"
