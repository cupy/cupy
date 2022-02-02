#!/bin/bash

# AUTO GENERATED: DO NOT EDIT!

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

export NVCC="ccache nvcc"

export CUPY_ACCELERATORS="cub,cutensor"

"$ACTIONS/build.sh"
"$ACTIONS/benchmark.sh"
"$ACTIONS/cleanup.sh"
