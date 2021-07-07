#!/bin/bash

set -uex

MARKER="${1:-}"

pytest_opts=(
    -rfEX
    --timeout 300
    --maxfail 500
    --showlocals
)

if [[ "${MARKER}" != "" ]]; then
    pytest_opts+=(-m "${MARKER}")
fi

python3 -m pip install --user pytest-timeout
python3 -m pip install --user -v ".[test]"

pushd tests
python3 -c 'import cupy; cupy.show_config()'
python3 -m pytest "${pytest_opts[@]}" .
popd
