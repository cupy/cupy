# Copied from cuda115.Dockerfile

ARG BASE_IMAGE="nvidia/cuda:11.5.0-devel-ubuntu20.04"
FROM ${BASE_IMAGE}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install \
       make build-essential libssl-dev zlib1g-dev \
       libbz2-dev libreadline-dev libsqlite3-dev wget \
       curl llvm libncursesw5-dev xz-utils tk-dev \
       libxml2-dev libxmlsec1-dev libffi-dev \
       liblzma-dev && \
    apt-get -qqy install ccache git curl && \
    apt-get -qqy --allow-change-held-packages \
            --allow-downgrades install 'libnccl2=2.11.*+cuda11.5' 'libnccl-dev=2.11.*+cuda11.5' 'libcutensor1=1.4.*' 'libcutensor-dev=1.4.*' 'libcusparselt0=0.2.0.*' 'libcusparselt-dev=0.2.0.*' 'libcudnn8=8.3.*+cuda11.5' 'libcudnn8-dev=8.3.*+cuda11.5'

ENV PATH="/usr/lib/ccache:${PATH}"

# Python 3.8 (Ubuntu 20.04)
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy install python3 python3-dev python3-pip python3-setuptools && \
    python3 -m pip install -U pip setuptools && \
    apt-get -qqy purge python3-pip python3-setuptools
