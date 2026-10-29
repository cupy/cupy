#!/bin/bash

set -uex

# Ubuntu 24.04 marks the system Python as PEP 668 externally-managed; this
# container runs as root with no venv, so let pip install into it directly.
export PIP_BREAK_SYSTEM_PACKAGES=1

# The ROCm math/comm libraries (hipblas, rccl, rocfft, etc.) are already
# shipped in the rocm/dev-ubuntu-24.04:*-full base image, so we only need to
# make sure Python's dev headers and pip are available. ca-certificates is
# listed explicitly so apt upgrades it. That covers apt, curl and git.
apt-get -y update
DEBIAN_FRONTEND=noninteractive apt-get -y install \
    ca-certificates curl python3-dev python3-wheel

# Ubuntu's python3-pip (24.0) trusts the compromised eMudhra/emSign CAs, so
# drop it if the base image has it and install pip from a hash-checked wheel
# run as its own installer. Keep the URL in sync with
# .github/actions/harden-pip/action.yml.
DEBIAN_FRONTEND=noninteractive apt-get -y purge python3-pip
pip_whl=/tmp/pip-26.2.2-py3-none-any.whl
curl -fsSLo "${pip_whl}" https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl
echo "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70  ${pip_whl}" | sha256sum -c -
python3 "${pip_whl}/pip" install --disable-pip-version-check "${pip_whl}"
rm "${pip_whl}"
python3 -m pip install --disable-pip-version-check --upgrade pip

hipconfig

export ROCM_HOME="/opt/rocm"
export HCC_AMDGPU_TARGET="gfx900"
export CUPY_INSTALL_USE_HIP="1"
python3 -m pip install -v -e .
python3 -c "import cupy; cupy.show_config()"
