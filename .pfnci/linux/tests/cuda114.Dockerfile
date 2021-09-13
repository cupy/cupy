# AUTO GENERATED: DO NOT EDIT!
ARG BASE_IMAGE="nvidia/cuda:11.4.0-devel-ubuntu20.04"
FROM ${BASE_IMAGE}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy --allow-change-held-packages install libnccl2=2.11.*+cuda11.4 libnccl-dev=2.11.*+cuda11.4 libcudnn8-dev=8.2.*+cuda11.4 libcutensor-dev=1.3.* libcusparselt-dev=0.1.* && \
    apt-get -qqy install ccache

ENV PATH "/usr/lib/ccache:${PATH}"

RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.9.6 && \
    pyenv global 3.9.6 && \
    pip install -U setuptools pip

RUN pip install -U numpy==1.21.* scipy==1.7.* optuna==2.* cython==0.29.*
