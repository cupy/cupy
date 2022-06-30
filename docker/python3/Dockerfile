FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir -U install setuptools pip
RUN pip3 install --no-cache-dir -f https://github.com/cupy/cupy/releases/v11.0.0rc1 "cupy-cuda11x[all]==11.0.0rc1"
