#!/bin/bash

set -uex


git clone https://github.com/cupy/cupy-performance.git performance
# TODO(ecastill): make this optional
pip install seaborn
ls
pushd /perf-results
# python prof.py benchmarks/bench_ufunc_cupy.py -c
# Concat all results
# Upload result to chainer-public
# gsutil -m -q cp *.csv "gs://chainer-artifacts-pfn-public-ci/cupy-ci/${CI_JOB_ID}/"
echo "test" > test.csv
popd

# case ${test_retval} in
#     0 )
#         echo "Result: SUCCESS"
#         ;;
#     124 )
#         echo "Result: TIMEOUT (INT)"
#         exit $test_retval
#         ;;
#     137 )
#         echo "Result: TIMEOUT (KILL)"
#         exit $test_retval
#         ;;
#     * )
#         echo "Result: FAIL ($test_retval)"
#         exit $test_retval
#         ;;
# esac
