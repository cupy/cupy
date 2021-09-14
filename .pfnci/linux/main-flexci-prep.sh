#!/bin/bash

# Build and push Docker images for CUDA test targets.

set -ue

env

for DF in "$(dirname ${0})"/tests/cuda*.Dockerfile; do
    echo "$(basename "${DF}")" | cut -d . -f 1
done | xargs -i -n 1 -P 4 bash -c 'echo "flexci-prep: $2 start"; "$1" "$2" build push rmi > /dev/null 2>&1 && echo "flexci-prep: $2 succeeded" || echo "flexci-prep: $2 failed"' -- "$(dirname ${0})/run.sh" "{}"
