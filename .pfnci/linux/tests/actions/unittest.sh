#!/bin/bash

set -uex

MARKER="${1:-}"; shift
PYTEST_FILES=(${@:-.})

pytest_opts=(
    -rfEX
    --timeout 300
    --maxfail 500
    --showlocals
    --durations 10
)

if [[ "${MARKER}" != "" ]]; then
    pytest_opts+=(-m "${MARKER}")
fi

# TODO: support coverage reporting
python3 -m pip install --user pytest-timeout pytest-xdist

pushd tests
timeout --signal INT --kill-after 10 60 python3 -c 'import cupy; cupy.show_config(_full=True)'
test_retval=0
timeout --signal INT --kill-after 60 36000 python3 -m pytest "${pytest_opts[@]}" "${PYTEST_FILES[@]}" || test_retval=$?
popd

case ${test_retval} in
    0 )
        echo "Result: SUCCESS"
        ;;
    124 )
        echo "Result: TIMEOUT (INT)"
        exit $test_retval
        ;;
    137 )
        echo "Result: TIMEOUT (KILL)"
        exit $test_retval
        ;;
    * )
        echo "Result: FAIL ($test_retval)"
        exit $test_retval
        ;;
esac
