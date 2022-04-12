FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir -U install setuptools pip
RUN pip3 install --no-cache-dir -f https://github.com/cupy/cupy/releases/v11.0.0b1 "cupy-cuda114[all]==11.0.0b1"
