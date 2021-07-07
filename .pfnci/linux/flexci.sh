#!/bin/bash

# Bootstrap script for FlexCI.

set -uex

TARGET="${1}"

env

stages=(cache_get test)
if [[ "${FLEXCI_BRANCH:-}" != refs/pull/* ]]; then
    # Cache will not be uploaded when testing pull-requests.
    stages+=(cache_put)
fi
"$(dirname ${0})/run.sh" "${TARGET}" "${stages[@]}"
