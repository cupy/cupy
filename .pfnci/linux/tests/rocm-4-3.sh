#!/bin/bash

# AUTO GENERATED: DO NOT EDIT!

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

# TODO(kmaehashi): Tentatively sparsen parameterization to make test run complete.
export CUPY_TEST_FULL_COMBINATION="0"
export CUPY_INSTALL_USE_HIP=1

# Show detailed progress to diagnose abort issues
export PYTEST_ADDOPTS="-s -v"

"$ACTIONS/build.sh"
"$ACTIONS/unittest.sh" "not slow and not multi_gpu"
"$ACTIONS/cleanup.sh"
