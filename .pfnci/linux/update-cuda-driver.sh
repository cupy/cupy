#!/bin/bash

set -ue

if dpkg -s cuda-drivers-495 && ls /dev/nvidiactl ; then
    killall Xorg
    nvidia-smi -pm 0

    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
    apt-get purge -qqy "cuda-drivers*" "*nvidia*-495"
    apt-get install -qqy "cuda-drivers"

    modprobe -r nvidia_drm nvidia_uvm nvidia_modeset nvidia
    nvidia-smi -pm 1
    nvidia-smi
fi
