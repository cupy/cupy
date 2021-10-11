#!/bin/bash

# AUTO GENERATED: DO NOT EDIT!

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

export NVCC="ccache nvcc"

"$ACTIONS/build.sh"
"$ACTIONS/unittest.sh" "not slow"
"$ACTIONS/cleanup.sh"
