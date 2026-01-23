#!/bin/bash

# AUTO GENERATED: DO NOT EDIT!

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

hipconfig

# TODO(kmaehashi): Tentatively sparsen parameterization to make test run complete.
export CUPY_TEST_FULL_COMBINATION="0"
export CUPY_INSTALL_USE_HIP=1

echo "================ Environment Variables ================"
env
echo "======================================================="

"$ACTIONS/build.sh"
"$ACTIONS/unittest.sh" "not slow and not multi_gpu" cupy_tests/core_tests/test_bfloat16.py
"$ACTIONS/cleanup.sh"
