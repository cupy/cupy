#!/bin/bash

# Bootstrap script for FlexCI.

set -ue

TARGET="${1}"

echo "Environment Variables:"
env

stages=(cache_get test)
if [[ "${FLEXCI_BRANCH:-}" != refs/pull/* ]]; then
    # Cache will not be uploaded when testing pull-requests.
    stages+=(cache_put)
fi

pull_req=""
if [[ "${FLEXCI_BRANCH:-}" == refs/pull/* ]]; then
    # Extract pull-request ID
    pull_req="$(echo "${FLEXCI_BRANCH}" | cut -d/ -f3)"
    echo "Testing Pull-Request: #${pull_req}"
fi

echo "Starting: "${TARGET}" "${stages[@]}""
CACHE_DIR=/tmp/cupy_cache PULL_REQUEST="${pull_req}" "$(dirname ${0})/run.sh" "${TARGET}" "${stages[@]}" > /tmp/log.txt 2>&1 && echo Test succeeded!
test_retval=$?
echo "Exit with status ${test_retval}"

echo "Uploading the log..."
gsutil -m -q cp /tmp/log.txt "gs://chainer-artifacts-pfn-public-ci/cupy-ci/${CI_JOB_ID}/"

echo "****************************************************************************************************"
echo "Last 100 lines of the log:"
echo "****************************************************************************************************"
tail -n 100 /tmp/log.txt
echo "****************************************************************************************************"
echo "Full log is available at:"
echo "https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/${CI_JOB_ID}/log.txt"
echo "****************************************************************************************************"

# TODO: implement gitter notification

exit ${test_retval}
