#!/bin/bash

set -uex

# Python 3.8 (Ubuntu 20.04)
apt-get -y update
DEBIAN_FRONTEND=noninteractive apt-get -y install python3-pip python3-dev

hipconfig

pip3 install -U pip wheel
pip3 install cython

export ROCM_HOME="/opt/rocm"
export HCC_AMDGPU_TARGET="gfx900"
export CUPY_INSTALL_USE_HIP="1"
pip3 install -v -e .
python3 -c "import cupy; cupy.show_config()"
