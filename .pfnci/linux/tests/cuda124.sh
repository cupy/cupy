#!/bin/bash

# AUTO GENERATED: DO NOT EDIT!

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

nvidia-smi

export NVCC="ccache nvcc"

export CUPY_ACCELERATORS="cutensor,cub"

echo "================ Environment Variables ================"
env
echo "======================================================="


trap "$ACTIONS/cleanup.sh" EXIT
"$ACTIONS/fetch-wheel.sh" native
# Ensure matrix versions in case CuPy pins changed them (nightly testing).
python3 -m pip install --user -U 'numpy==2.2.*' 'scipy==1.16.*' 'optuna==3.*' 'cython==3.2.*,!=3.2.6'
CUPY_CI_PYTEST_EXTRA_OPTS="${CUPY_CI_PYTEST_EXTRA_OPTS:+$CUPY_CI_PYTEST_EXTRA_OPTS }--deselect tests/install_tests/test_cupy_builder/test_features.py::test_CUDA_cuda" "$ACTIONS/unittest.sh" "not slow and not multi_gpu"
