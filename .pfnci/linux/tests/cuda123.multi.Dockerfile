# AUTO GENERATED: DO NOT EDIT!
ARG BASE_IMAGE="nvidia/cuda:12.3.0-devel-ubuntu20.04"
FROM ${BASE_IMAGE}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install \
       make build-essential libssl-dev zlib1g-dev \
       libbz2-dev libreadline-dev libsqlite3-dev wget \
       curl llvm libncursesw5-dev xz-utils tk-dev \
       libxml2-dev libxmlsec1-dev libffi-dev \
       liblzma-dev \
       libopenmpi-dev \
       && \
    apt-get -qqy install ccache git curl && \
    apt-get -qqy --allow-change-held-packages \
            --allow-downgrades install 'libnccl2=2.19.*+cuda12.3' 'libnccl-dev=2.19.*+cuda12.3' 'libcutensor2=2.0.*' 'libcutensor-dev=2.0.*' 'libcusparselt0=0.2.0.*' 'libcusparselt-dev=0.2.0.*' 'libcudnn8=8.8.*+cuda12.0' 'libcudnn8-dev=8.8.*+cuda12.0'

ENV PATH "/usr/lib/ccache:${PATH}"

COPY setup/update-alternatives-cutensor.sh /
RUN /update-alternatives-cutensor.sh

RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.12.0 && \
    pyenv global 3.12.0 && \
    pip install -U setuptools pip wheel

RUN pip install -U 'numpy==1.26.*' 'scipy==1.11.*' 'optuna==3.*' 'mpi4py==3.*' 'cython==0.29.*'
RUN pip uninstall -y cuda-python && \
    pip check
