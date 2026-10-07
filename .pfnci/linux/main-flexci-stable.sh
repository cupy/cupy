#!/bin/bash

set -uex

BRANCH="$(< ./.pfnci/BRANCH_STABLE)"
git clone --branch "${BRANCH}" --depth 1 --recursive https://github.com/cupy/cupy.git "cupy-${BRANCH}"
cd "cupy-${BRANCH}"
# The stable branch predates the wheel-fetch token and its log redaction, and
# builds from source (no token needed). Drop the token before delegating so the
# stable job's env dump can't leak it.
unset CUPY_CI_GITHUB_TOKEN GH_TOKEN
./.pfnci/linux/main-flexci.sh "$@"
