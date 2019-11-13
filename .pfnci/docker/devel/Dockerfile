FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

ARG PYTHON
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python${PYTHON}-dev \
    $([ ${PYTHON} = 2.7 ] || echo python${PYTHON}-distutils) \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LO https://bootstrap.pypa.io/get-pip.py \
    && python${PYTHON} get-pip.py --no-cache-dir \
    && rm get-pip.py \
    && pip${PYTHON} install --no-cache-dir cython
