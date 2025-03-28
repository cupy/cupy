#!/bin/bash

echo "::group::Test - Build"
set -uex

time CUPY_NUM_NVCC_THREADS=5 CUPY_NUM_BUILD_JOBS=8 python3 -m pip install --user -v ".[test]"

set +uex
echo "::endgroup::"
