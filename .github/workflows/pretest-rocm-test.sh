#!/bin/bash

set -uex

hipconfig

pip install -U pip wheel
pip install cython

export HCC_AMDGPU_TARGET=gfx900
export CUPY_INSTALL_USE_HIP=1
pip install -v -e .
python -c "import cupy; cupy.show_config()"
