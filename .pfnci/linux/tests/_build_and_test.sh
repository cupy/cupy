#!/bin/bash

set -uex

MARKER="${1:-}"

pytest_opts=(
    -rfEX
    --timeout 300
    --maxfail 500
    --showlocals
    --numprocesses auto
)

if [[ "${MARKER}" != "" ]]; then
    pytest_opts+=(-m "${MARKER}")
fi

python3 -m pip install --user pytest-timeout pytest-xdist
python3 -m pip install --user -v ".[all,test]"

pushd tests
python3 -c 'import cupy; cupy.show_config()'
python3 -m pytest "${pytest_opts[@]}" .
popd

python3 .pfnci/trim_cupy_kernel_cache.py --max-size $((5*1024*1024*1024)) --rm
