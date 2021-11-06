#!/bin/bash

set -uex

time CUPY_NUM_NVCC_THREADS=8 CUPY_NUM_BUILD_JOBS=8 python3 -m pip install --user -v ".[test]"
