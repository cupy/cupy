#!/bin/bash

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

export NVCC="ccache nvcc"

export CUPY_USE_CUDA_PYTHON="1"

"$ACTIONS/build.sh"
"$ACTIONS/unittest.sh" "not slow"
"$ACTIONS/cleanup.sh"
