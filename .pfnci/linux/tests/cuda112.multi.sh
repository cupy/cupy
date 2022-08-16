#!/bin/bash

# AUTO GENERATED: DO NOT EDIT!

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

export NVCC="ccache nvcc"

export CUPY_ACCELERATORS=""

"$ACTIONS/build.sh"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
"$ACTIONS/unittest.sh" "not slow and multi_gpu"
"$ACTIONS/cleanup.sh"
