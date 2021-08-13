#!/bin/bash

set -uex

export NVCC="ccache nvcc"
export CUPY_USE_CUDA_PYTHON="1"

. "$(dirname $0)/_environment.sh"

python3 -m pip install --user Cython
python3 -m pip install --user https://github.com/NVIDIA/cuda-python/archive/refs/heads/master.tar.gz

bash "$(dirname $0)/_build_and_test.sh" "not slow"
