#!/bin/bash

# AUTO GENERATED: DO NOT EDIT!

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

nvidia-smi

export NVCC="ccache nvcc"

export CUPY_ACCELERATORS="cutensor,cub"

echo "================ Environment Variables ================"
env
echo "======================================================="

"$ACTIONS/build.sh"
"$ACTIONS/benchmark.sh"
"$ACTIONS/cleanup.sh"
