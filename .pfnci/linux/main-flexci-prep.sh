#!/bin/bash

# Build and push Docker images for all test targets.

set -ue

env

gcloud auth configure-docker asia-northeast1-docker.pkg.dev

for DF in $(find "$(dirname ${0})/tests" -name "*.Dockerfile" -type f); do
    echo "$(basename "${DF}" .Dockerfile)"
done | xargs -i -n 1 -P 4 bash -c '
    echo "flexci-prep: $2 start";
    DOCKER_IMAGE_CACHE=0 "$1" "$2" build push rmi > "log_$2.out" 2>&1 && (
        echo "flexci-prep: $2 succeeded"
    ) || (
        echo "flexci-prep: $2 *** FAILED ***" && cat "log_$2.out" && echo "flexci-prep: terminating further builds" && exit 255
    )
    ' -- "$(dirname ${0})/run.sh" "{}"
