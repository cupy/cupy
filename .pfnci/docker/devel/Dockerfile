FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python-dev \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LO https://bootstrap.pypa.io/get-pip.py \
    && python2.7 get-pip.py --no-cache-dir \
    && python3.6 get-pip.py --no-cache-dir \
    && rm get-pip.py

RUN pip2.7 install --no-cache-dir cython \
    && pip3.6 install --no-cache-dir cython
