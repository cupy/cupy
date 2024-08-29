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

###
export CUPY_CACHE_DIR=/tmp/this_pr_only
export CUPY_CUDA_COMPILE_WITH_DEBUG=1
mkdir -p "${CUPY_CACHE_DIR}"
compute-sanitizer --tool memcheck --padding 16 $(pyenv which python3) -m pytest "${pytest_opts[@]}" -v -k 'test_alternative_call[True-nearest]' cupyx_tests/scipy_tests/interpolate_tests/test_ndgriddata.py || test_retval=$?
compute-sanitizer --tool memcheck --padding 16 $(pyenv which python3) -m pytest "${pytest_opts[@]}" -v -k 'test_alternative_call[True-linear]' cupyx_tests/scipy_tests/interpolate_tests/test_ndgriddata.py || test_retval=$?
compute-sanitizer --tool memcheck --padding 16 $(pyenv which python3) -m pytest "${pytest_opts[@]}" -v -k 'test_alternative_call[True-cubic]' cupyx_tests/scipy_tests/interpolate_tests/test_ndgriddata.py || test_retval=$?
compute-sanitizer --tool memcheck --padding 16 $(pyenv which python3) -m pytest "${pytest_opts[@]}" -v -k 'test_alternative_call[False-nearest]' cupyx_tests/scipy_tests/interpolate_tests/test_ndgriddata.py || test_retval=$?
compute-sanitizer --tool memcheck --padding 16 $(pyenv which python3) -m pytest "${pytest_opts[@]}" -v -k 'test_alternative_call[False-linear]' cupyx_tests/scipy_tests/interpolate_tests/test_ndgriddata.py || test_retval=$?
compute-sanitizer --tool memcheck --padding 16 $(pyenv which python3) -m pytest "${pytest_opts[@]}" -v -k 'test_alternative_call[False-cubic]' cupyx_tests/scipy_tests/interpolate_tests/test_ndgriddata.py || test_retval=$?
exit 1
###

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
