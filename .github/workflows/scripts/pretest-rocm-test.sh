#!/bin/bash

set -uex

# Ubuntu 24.04 marks the system Python as PEP 668 externally-managed; this
# container runs as root with no venv, so let pip install into it directly.
export PIP_BREAK_SYSTEM_PACKAGES=1

# The ROCm math/comm libraries (hipblas, rccl, rocfft, etc.) are already
# shipped in the rocm/dev-ubuntu-24.04:*-full base image, so we only need to
# make sure Python's dev headers and pip are available. ca-certificates is
# listed explicitly so apt upgrades it. That covers apt, curl and git.
# noble's python3-pip is 24.0 and trusts only its vendored certifi, so pip
# itself is bootstrapped below.
apt-get -y update
DEBIAN_FRONTEND=noninteractive apt-get -y install \
    ca-certificates python3-dev python3-pip python3-wheel

# Remove the pinned pip once Ubuntu's python3-pip is >= 26.2.2
# (see .github/actions/harden-pip).
python3 -m pip install --disable-pip-version-check "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
python3 -m pip install --disable-pip-version-check --upgrade pip

hipconfig

export ROCM_HOME="/opt/rocm"
export HCC_AMDGPU_TARGET="gfx900"
export CUPY_INSTALL_USE_HIP="1"
python3 -m pip install -v -e .
python3 -c "import cupy; cupy.show_config()"
