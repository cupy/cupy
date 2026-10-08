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
python3 -m pip install --user -U 'numpy==2.0.*' 'scipy==1.14.*' 'optuna==3.*' 'mpi4py==4.*' 'ml_dtypes==0.5.*' 'cython==3.2.*,!=3.2.6'
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
CUPY_CI_PYTEST_EXTRA_OPTS="${CUPY_CI_PYTEST_EXTRA_OPTS:+$CUPY_CI_PYTEST_EXTRA_OPTS }--deselect tests/install_tests/test_cupy_builder/test_features.py::test_CUDA_cuda" "$ACTIONS/unittest.sh" "not slow and multi_gpu"
