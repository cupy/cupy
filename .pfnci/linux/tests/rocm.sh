#!/bin/bash

set -uex

export CUPY_INSTALL_USE_HIP=1

. "$(dirname $0)/_environment.sh"
bash "$(dirname $0)/_build_and_test.sh" "not slow"
