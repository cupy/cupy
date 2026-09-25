#!/bin/bash

# Bootstrap script for FlexCI.

set -ue

TARGET="${1}"
LOG_FILE="/tmp/log.txt"

echo "Environment Variables:"
# Redact the wheel-fetch token (provisioned in the FlexCI job env) from the
# published log; fetch-wheel.sh receives it via a mounted file, not the env.
env | grep -v -e '^CUPY_CI_GITHUB_TOKEN=' -e '^GH_TOKEN='

pull_req=""
if [[ "${FLEXCI_BRANCH:-}" == refs/pull/* ]]; then
    # Extract pull-request ID
    pull_req="$(echo "${FLEXCI_BRANCH}" | cut -d/ -f3)"
    echo "Testing Pull-Request: #${pull_req}"
fi

.pfnci/linux/update-cuda-driver.sh

gcloud auth configure-docker asia-northeast1-docker.pkg.dev

echo "Starting: "${TARGET}""
echo "****************************************************************************************************"

STAGES="cache_get build test"
if [[ "${TARGET}" == "benchmark" ]]; then
    STAGES="cache_get build benchmark"
fi
JUNIT_DIR=/tmp/cupy_junit
mkdir -p "${JUNIT_DIR}"
rm -f "${JUNIT_DIR}/junit.xml"
BENCHMARK_DIR=/tmp/benchmark CACHE_DIR=/tmp/cupy_cache JUNIT_DIR="${JUNIT_DIR}" CACHE_KERNEL_TO_GCS=1 PULL_REQUEST="${pull_req}" "$(dirname ${0})/run.sh" "${TARGET}" "${STAGES}" 2>&1 | tee "${LOG_FILE}"
test_retval=${PIPESTATUS[0]}

echo "****************************************************************************************************"
echo "Build & Test: Exit with status ${test_retval}"

if [[ "${pull_req}" == "" ]]; then
    # Upload cache when testing a branch, even when test failed.
    echo "Uploading cache and Docker image..."
    CACHE_DIR=/tmp/cupy_cache PULL_REQUEST="${pull_req}" "$(dirname ${0})/run.sh" "${TARGET}" cache_put push | tee --append "${LOG_FILE}"
    echo "Upload: Exit with status ${PIPESTATUS[0]}"

    # Notify.
    if [[ ${test_retval} != 0 ]]; then
        pip3 install -q slack-sdk gitterpy
        ./.pfnci/flexci_notify.py "TEST FAILED"
    fi
else
    # Upload cache when testing a PR.
    echo "Uploading cache..."
    CACHE_DIR=/tmp/cupy_cache PULL_REQUEST="${pull_req}" "$(dirname ${0})/run.sh" "${TARGET}" cache_put | tee --append "${LOG_FILE}"
    echo "Upload: Exit with status ${PIPESTATUS[0]}"
fi

echo "Uploading the log..."
gsutil -m -q cp "${LOG_FILE}" "gs://chainer-artifacts-pfn-public-ci/cupy-ci/${CI_JOB_ID}/"

junit_url=""
if [[ -f "${JUNIT_DIR}/junit.xml" ]]; then
    gzip -4 -c "${JUNIT_DIR}/junit.xml" > "${JUNIT_DIR}/junit.xml.gz"
    gsutil -m -q cp "${JUNIT_DIR}/junit.xml.gz" "gs://chainer-artifacts-pfn-public-ci/cupy-ci/${CI_JOB_ID}/"
    junit_url="https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/${CI_JOB_ID}/junit.xml.gz"
fi

echo "****************************************************************************************************"
echo "Full log is available at:"
echo "https://storage.googleapis.com/chainer-artifacts-pfn-public-ci/cupy-ci/${CI_JOB_ID}/log.txt"
if [[ -n "${junit_url}" ]]; then
    echo "JUnit XML is available at:"
    echo "${junit_url}"
fi
echo "****************************************************************************************************"

exit ${test_retval}
