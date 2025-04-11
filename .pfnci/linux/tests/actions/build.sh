#!/bin/bash

set -uex

# Use the "--no-build-isolation" option to test building with dependencies
# installed in the current environment. Otherwise builds always run in an
# isolated environment with the latest version of the build-system
# requirements specified in pyproject.toml and `setup_requires`.
time CUPY_NUM_NVCC_THREADS=5 CUPY_NUM_BUILD_JOBS=8 python3 -m pip install --no-build-isolation --user -v ".[test]"
