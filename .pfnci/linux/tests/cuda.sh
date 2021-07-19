#!/bin/bash

set -uex

export NVCC="ccache nvcc"

. "$(dirname $0)/_environment.sh"
bash "$(dirname $0)/_build_and_test.sh" "not slow"
