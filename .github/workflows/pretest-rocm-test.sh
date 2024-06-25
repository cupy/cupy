#!/bin/bash

set -uex

apt-get -y update
DEBIAN_FRONTEND=noninteractive apt-get -y install python3.9-dev python3-pip

hipconfig

python3.9 -m pip install -U pip wheel

pip install git+https://github.com/ROCmSoftwarePlatform/hipify_torch.git

export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME="/opt/rocm"
export HCC_AMDGPU_TARGET="gfx908,gfx90a,gfx940,gfx941,gfx942"
export CUPY_INSTALL_USE_HIP="1"
python3.9 -m pip install -v -e .
python3.9 -c "import cupy; cupy.show_config()"
