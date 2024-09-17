#!/bin/bash

set -ue

echo "Installed cuda-drivers:"
dpkg -l | grep cuda-drivers

# If CUDA driver of this version is installed, upgrade to the latest one.
CUDA_DRIVER_VERSION=525

if dpkg -s "cuda-drivers-${CUDA_DRIVER_VERSION}" && ls /dev/nvidiactl ; then
    killall Xorg || true
    nvidia-smi -pm 0

    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
    apt-get purge -qqy "cuda-drivers*" "*nvidia*-${CUDA_DRIVER_VERSION}"
    apt-get install -qqy "cuda-drivers"

    modprobe -r nvidia_drm nvidia_uvm nvidia_modeset nvidia
    nvidia-smi -pm 1
    nvidia-smi
fi
