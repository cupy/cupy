#!/bin/bash

set -uex

# The ROCm math/comm libraries (hipblas, rccl, rocfft, etc.) are already
# shipped in the rocm/dev-ubuntu-22.04:*-full base image, so we only need to
# make sure Python's dev headers and pip are available.
apt-get -y update
DEBIAN_FRONTEND=noninteractive apt-get -y install \
    python3-dev python3-pip

hipconfig

python3 -m pip install -U pip wheel

export ROCM_HOME="/opt/rocm"
export HCC_AMDGPU_TARGET="gfx900"
export CUPY_INSTALL_USE_HIP="1"
python3 -m pip install -v -e .
python3 -c "import cupy; cupy.show_config()"
