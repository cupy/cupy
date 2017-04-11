FROM nvidia/cuda:7.5-cudnn5-devel

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python-dev \
    python-pip && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip install chainer==1.23.0
