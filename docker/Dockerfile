FROM nvidia/cuda:7.5-cudnn5-devel

RUN apt-get update -y && \
    apt-get install -y \
    python-dev \
    python-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install chainer
