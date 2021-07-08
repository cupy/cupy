#!/bin/bash

# Bootstrap script for FlexCI.

set -uex

TARGET="${1}"
export CACHE_DIR=/tmp/cupy_cache

env

stages=(cache_get test)
if [[ "${FLEXCI_BRANCH:-}" != refs/pull/* ]]; then
    # Cache will not be uploaded when testing pull-requests.
    stages+=(cache_put)
fi

"$(dirname ${0})/run.sh" "${TARGET}" "${stages[@]}" > /tmp/log.txt 2>&1 && echo Test succeeded!
test_retval=$?

gsutil -m -q cp /tmp/log.txt "gs://chainer-artifacts-pfn-public-ci/cupy-ci/${CI_JOB_ID}/"
echo "Full log is available at: https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/${CI_JOB_ID}/log.txt"
echo "Last 100 lines of the log:"
tail -n 100 /tmp/log.txt

exit ${test_retval}
