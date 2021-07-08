#!/bin/bash

set -uex

. "$(dirname $0)/_environment.sh"

export CUPY_INSTALL_USE_HIP=1
#export NVCC="ccache nvcc"

hipconfig
env

python3 -m pip install --user pytest-timeout
python3 -m pip install --user -v ".[test]"

pushd tests
python3 -c 'import cupy; cupy.show_config()'
python3 -m pytest \
    -rfEX \
    --timeout 300 \
    --maxfail 500 \
    --showlocals \
    -m 'not slow' \
    .
popd
