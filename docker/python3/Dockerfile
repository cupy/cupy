FROM nvidia/cuda:9.2-cudnn7-devel

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir cupy-cuda92==6.0.0b3
