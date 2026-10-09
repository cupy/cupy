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
"$ACTIONS/fetch-wheel.sh"
# Ensure matrix versions in case CuPy pins changed them (nightly testing).
python3 -m pip install --user -U 'numpy==2.4.*' 'scipy==1.16.*' 'optuna==4.*' 'ml_dtypes==0.5.*' 'cython==3.2.*,!=3.2.6'
"$ACTIONS/benchmark.sh"
