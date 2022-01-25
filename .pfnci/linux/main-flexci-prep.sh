#!/bin/bash

# Build and push Docker images for all test targets.

set -ue

env

gcloud auth configure-docker

for DF in "$(dirname ${0})"/tests/*.Dockerfile; do
    echo "$(basename "${DF}")" | cut -d . -f 1
done | xargs -i -n 1 -P 4 bash -c 'echo "flexci-prep: $2 start"; DOCKER_IMAGE_CACHE=0 "$1" "$2" build push rmi > /dev/null 2>&1 && echo "flexci-prep: $2 succeeded" || echo "flexci-prep: $2 failed"' -- "$(dirname ${0})/run.sh" "{}"
