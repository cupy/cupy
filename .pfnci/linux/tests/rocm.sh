#!/bin/bash

set -uex

# TODO(kmaehashi): Tentatively sparsen parameterization to make test run complete.
export CUPY_TEST_FULL_COMBINATION="0"
export CUPY_INSTALL_USE_HIP=1

. "$(dirname $0)/_environment.sh"
bash "$(dirname $0)/_build_and_test.sh" "not slow"
