FROM nvidia/cuda:8.0-cudnn6-devel

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python-dev \
    python-pip \
    python-wheel \
    python-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip install --no-cache-dir cupy-cuda80==5.0.0b2
