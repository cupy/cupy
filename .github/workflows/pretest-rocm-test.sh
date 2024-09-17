#!/bin/bash

set -uex

apt-get -y update
DEBIAN_FRONTEND=noninteractive apt-get -y install python3-dev python3-pip

hipconfig

python3 -m pip install -U pip wheel

export ROCM_HOME="/opt/rocm"
export HCC_AMDGPU_TARGET="gfx900"
export CUPY_INSTALL_USE_HIP="1"
python3 -m pip install -v -e .
python3 -c "import cupy; cupy.show_config()"
