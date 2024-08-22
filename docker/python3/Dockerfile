FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir -U setuptools pip
RUN pip3 install --no-cache-dir "cupy-cuda12x[all]==13.3.0"
