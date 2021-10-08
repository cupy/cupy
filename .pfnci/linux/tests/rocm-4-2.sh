#!/bin/bash

# AUTO GENERATED: DO NOT EDIT!

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

export CUPY_INSTALL_USE_HIP=1
export AMD_LOG_LEVEL=4

"$ACTIONS/build.sh"
"$ACTIONS/unittest.sh" "not slow"
"$ACTIONS/cleanup.sh"
