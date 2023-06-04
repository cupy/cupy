# AUTO GENERATED: DO NOT EDIT!
ARG BASE_IMAGE="nvidia/cuda:10.2-devel-ubuntu18.04"
FROM ${BASE_IMAGE}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install software-properties-common && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/ /"

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
            --allow-downgrades install 'libnccl2=2.8.*+cuda10.2' 'libnccl-dev=2.8.*+cuda10.2' 'libcudnn7=7.6.*+cuda10.2' 'libcudnn7-dev=7.6.*+cuda10.2'

ENV PATH "/usr/lib/ccache:${PATH}"

RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.8.11 && \
    pyenv global 3.8.11 && \
    pip install -U setuptools pip wheel

RUN pip install -U 'numpy==1.20.*' 'scipy==1.6.*' 'optuna==3.*' 'mpi4py==3.*' 'cython==0.29.*'
RUN pip uninstall -y cuda-python && \
    pip check
