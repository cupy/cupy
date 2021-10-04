#!/bin/bash

set -uex

MARKER="${1:-}"

pytest_opts=(
    -rfEX
    --timeout 300
    --maxfail 500
    --showlocals
    --numprocesses 2
)

if [[ "${MARKER}" != "" ]]; then
    pytest_opts+=(-m "${MARKER}")
fi

# TODO: support coverage reporting
python3 -m pip install --user pytest-timeout pytest-xdist

pushd tests
python3 -c 'import cupy; cupy.show_config()'
python3 -m pytest "${pytest_opts[@]}" .
popd
