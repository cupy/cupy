#!/bin/bash

# Bootstrap script for GitHub Actions Self-Hosted CI.

set -ue

TARGET="${1}"

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

STAGES="build test"
CACHE_DIR=/tmp/cupy_cache PULL_REQUEST="${pull_req}" "$(dirname ${0})/run.sh" "${TARGET}" "${STAGES}"
test_retval=$?

echo "****************************************************************************************************"
echo "Build & Test: Exit with status ${test_retval}"

if [[ "${pull_req}" == "" ]]; then
    # Notify.
    if [[ ${test_retval} != 0 ]]; then
        pip3 install -q slack-sdk gitterpy
        ./.pfnci/flexci_notify.py "TEST FAILED"
    fi
fi

exit ${test_retval}
