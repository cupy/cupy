#!/bin/bash

set -uex

BRANCH="$(< ./.pfnci/BRANCH_STABLE)"
git clone --branch "${BRANCH}" --depth 1 --recursive https://github.com/cupy/cupy.git "cupy-${BRANCH}"
cd "cupy-${BRANCH}"
./.pfnci/linux/main-flexci.sh "$@"
