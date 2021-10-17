#!/bin/bash

# Bootstrap script for FlexCI.

set -ue

TARGET="${1}"
LOG_FILE="/tmp/log.txt"

echo "Environment Variables:"
env

pull_req=""
if [[ "${FLEXCI_BRANCH:-}" == refs/pull/* ]]; then
    # Extract pull-request ID
    pull_req="$(echo "${FLEXCI_BRANCH}" | cut -d/ -f3)"
    echo "Testing Pull-Request: #${pull_req}"
fi

echo "Starting: "${TARGET}""
echo "****************************************************************************************************"
CACHE_DIR=/tmp/cupy_cache PULL_REQUEST="${pull_req}" "$(dirname ${0})/run.sh" "${TARGET}" cache_get build test 2>&1 | tee "${LOG_FILE}"
test_retval=${PIPESTATUS[0]}
echo "****************************************************************************************************"
echo "Exit with status ${test_retval}"

if [[ "${pull_req}" == "" ]]; then
    # Upload cache when testing a branch, even when test failed.
    echo "Uploading cache and Docker image..."
    CACHE_DIR=/tmp/cupy_cache PULL_REQUEST="${pull_req}" "$(dirname ${0})/run.sh" "${TARGET}" cache_put push | tee --append "${LOG_FILE}"
    echo "Upload exit with status ${PIPESTATUS[0]}"

    # Notify.
    if [[ ${test_retval} != 0 ]]; then
        pip install -q slack-sdk gitterpy
        ./.pfnci/flexci_notify.py "Test failed."
    fi
fi

echo "Uploading the log..."
gsutil -m -q cp "${LOG_FILE}" "gs://chainer-artifacts-pfn-public-ci/cupy-ci/${CI_JOB_ID}/"

echo "****************************************************************************************************"
echo "Full log is available at:"
echo "https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/${CI_JOB_ID}/log.txt"
echo "****************************************************************************************************"

exit ${test_retval}
