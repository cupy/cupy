FROM golang AS xpytest
RUN git clone --depth=1 https://github.com/chainer/xpytest.git /xpytest
RUN cd /xpytest && \
    go build -o /usr/local/bin/xpytest ./cmd/xpytest

FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python-dev python-pip python-wheel python-setuptools \
    python3-dev python3-pip python3-wheel python3-setuptools \
    wget git g++ make && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN python3.5 -m pip install --upgrade pip setuptools
RUN python2.7 -m pip install --upgrade pip setuptools

RUN python3.5 -m pip install \
    'cython>=0.28.0' 'pytest==4.1.1' 'pytest-xdist==1.26.1' \
    mock setuptools filelock 'numpy>=1.9.0' 'protobuf>=3.0.0' 'six>=1.9.0'

RUN python2.7 -m pip install \
    'cython>=0.28.0' 'pytest==4.1.1' 'pytest-xdist==1.26.1' \
    mock setuptools filelock 'protobuf>=3.0.0' 'six>=1.9.0'

# Newer versions of numpy and more-itertools no longer support python2.
RUN python2.7 -m pip install 'numpy<=1.16.5' 'more-itertools<=5.0.0'

COPY --from=xpytest /usr/local/bin/xpytest /usr/local/bin/xpytest
