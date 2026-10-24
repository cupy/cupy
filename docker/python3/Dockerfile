FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir --break-system-packages -f https://pip.cupy.dev/pre/ "cupy-cuda13x[all]==14.0.0rc1"
