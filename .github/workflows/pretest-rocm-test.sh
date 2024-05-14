#!/bin/bash

set -uex
export DEBIAN_FRONTEND=noninteractive

# Add the deadsnakes PPA and update package lists
apt-get update && apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt-get update

# Install necessary tools and dependencies
apt-get install -y python3.9-dev python3-pip
python3.9 -m pip install -U pip wheel

# Install hipify-torch (pin for stability) 
pip install git+https://github.com/ROCmSoftwarePlatform/hipify_torch.git@cbce9fbfd2783e932a45e3b47ca522e59b880c66

# Display ROCm configuration/environment information
hipconfig

# Setup build env on ROCm
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME="/opt/rocm"
export HCC_AMDGPU_TARGET="gfx908,gfx90a,gfx940,gfx941,gfx942"
export CUPY_INSTALL_USE_HIP="1"

# Build CuPy on ROCm
python3.9 -m pip install -v -e .
python3.9 -c "import cupy; cupy.show_config()"
