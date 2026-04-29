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

if [[ "${MARKER}" =~ "not slow" ]]; then
    # I we are not running slow tests (and not high memory)
    # running tests in parallel should be safe.
    pytest_opts+=("-n2")
    # Use nvidia-cuda-mps-control -d to start MPS server and hopefully help execution speed on T4
    nvidia-cuda-mps-control -d
fi

# TODO: support coverage reporting
python3 -m pip install --user pytest-timeout pytest-xdist

pushd tests
timeout --signal INT --kill-after 10 60 python3 -c 'import cupy; cupy.show_config(_full=True)'
test_retval=0
timeout --signal INT --kill-after 60 18000 python3 -m pytest "${pytest_opts[@]}" "${PYTEST_FILES[@]}" || test_retval=$?
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

# Validate that importing cupy does not import cupyx
# grep returns 0 if it finds anything so invert the result.
! $(python3 -Ximporttime -c "import cupy" |& grep -q cupyx) || (echo "Found forbidden import 'cupyx'" && exit 1)
