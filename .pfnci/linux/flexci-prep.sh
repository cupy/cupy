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
for P in ${WAIT_PIDS}; do
    wait ${P} || echo "flexci-prep: Build failed for PID ${P}"
done
