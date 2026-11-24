#!/bin/bash

# AUTO GENERATED: DO NOT EDIT!

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

nvidia-smi

export NVCC="ccache nvcc"

export CUPY_ACCELERATORS="cub,cutensor"

echo "================ Environment Variables ================"
env
echo "======================================================="


trap "$ACTIONS/cleanup.sh" EXIT
"$ACTIONS/fetch-wheel.sh"
# Ensure matrix versions in case CuPy pins changed them (nightly testing).
python3 -m pip install --user -U --extra-index-url=https://pypi.anaconda.org/scientific-python-nightly-wheels/simple 'numpy>=0a0' 'scipy>=0a0,<2.0' 'optuna>=0a0' 'ml_dtypes>=0a0' 'cython==3.2.*,!=3.2.6'
"$ACTIONS/array-api.sh"
