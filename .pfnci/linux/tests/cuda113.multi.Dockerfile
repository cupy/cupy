# AUTO GENERATED: DO NOT EDIT!
ARG BASE_IMAGE="nvidia/cuda:11.3.1-devel-ubuntu18.04"
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
            --allow-downgrades install 'libnccl2=2.9.*+cuda11.3' 'libnccl-dev=2.9.*+cuda11.3' 'libcutensor1=1.5.*' 'libcutensor-dev=1.5.*' 'libcudnn8=8.2.*+cuda11.4' 'libcudnn8-dev=8.2.*+cuda11.4'

ENV PATH "/usr/lib/ccache:${PATH}"

COPY setup/update-alternatives-cutensor.sh /
RUN /update-alternatives-cutensor.sh

RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.8.11 && \
    pyenv global 3.8.11 && \
    pip install -U setuptools pip wheel

RUN pip install -U 'numpy==1.20.*' 'scipy==1.6.*' 'optuna==3.*' 'mpi4py==3.*' 'cython==0.29.*'
RUN pip uninstall -y cuda-python && \
    pip check
