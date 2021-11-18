# AUTO GENERATED: DO NOT EDIT!
ARG BASE_IMAGE="nvidia/cuda:10.2-devel-ubuntu18.04"
FROM ${BASE_IMAGE}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install \
       make build-essential libssl-dev zlib1g-dev \
       libbz2-dev libreadline-dev libsqlite3-dev wget \
       curl llvm libncursesw5-dev xz-utils tk-dev \
       libxml2-dev libxmlsec1-dev libffi-dev \
       liblzma-dev && \
    apt-get -qqy install ccache git curl && \
    apt-get -qqy --allow-change-held-packages \
            --allow-downgrades install libnccl2=2.8.*+cuda10.2 libnccl-dev=2.8.*+cuda10.2 libcudnn7=7.6.*+cuda10.2 libcudnn7-dev=7.6.*+cuda10.2

ENV PATH "/usr/lib/ccache:${PATH}"

RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.7.11 && \
    pyenv global 3.7.11 && \
    pip install -U setuptools pip

RUN pip install -U numpy==1.18.* scipy==1.4.* optuna==2.* cython==0.29.*

ENV HOME /tmp
