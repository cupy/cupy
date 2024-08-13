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

ROCM_TEST_SCRIPT_DIR=$(dirname "$(realpath "$0")")

pushd tests
timeout --signal INT --kill-after 10 60 python3 -c 'import cupy; cupy.show_config(_full=True)'
test_retval=0
if [[ -f /opt/rocm/bin/rocm-smi ]]; then
    # setup pytest_opts as string to pass to rocm test script
    # Add another set of quotes around MARKER
    index=$(( ${#pytest_opts[@]} - 1 ))
    pytest_opts[$index]='"'"${pytest_opts[$index]}"'"'

    pytest_opts_str="${pytest_opts[@]}"

    # ROCm test script currently doesn't support running partial tests
    timeout --signal INT --kill-after 60 18000 python3 "${ROCM_TEST_SCRIPT_DIR}/run_tests_rocm.py" --pytest-opts "${pytest_opts_str}" --all-tests || test_retval=$?
else
    timeout --signal INT --kill-after 60 18000 python3 -m pytest "${pytest_opts[@]}" "${PYTEST_FILES[@]}" || test_retval=$?
fi
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
