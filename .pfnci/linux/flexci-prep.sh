#!/bin/bash

# Build and push Docker images for all test targets.

set -ue

env

WAIT_PIDS=""
for DF in "$(dirname ${0})"/tests/*.Dockerfile; do
    TARGET="$(echo "$(basename "${DF}")" | cut -d . -f 1)"
    "$(dirname ${0})/run.sh" "${TARGET}" build push &
    docker_pid=$!
    echo "flexci-prep: Building docker image for target $TARGET (PID: ${docker_pid})"
    WAIT_PIDS="${WAIT_PIDS} ${docker_pid}"
done

# Wait until the build and push complete.
FAILURES=0
for P in ${WAIT_PIDS}; do
    WAIT_RESULT=0
    wait ${P} || WAIT_RESULT=$?
    if [[ "${WAIT_RESULT}" != "0" ]]; then
        echo "flexci-prep: Build failed for PID ${P}"
        FAILURES="$((${FAILURES} + 1))"
    fi
done

if [[ "${FAILURES}" != "0" ]]; then
    echo "flexci-prep: ${FAILURES} failures"
    exit 1
fi
