# AUTO GENERATED: DO NOT EDIT!
ARG BASE_IMAGE="nvidia/cuda:11.8.0-devel-ubuntu20.04"
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
            --allow-downgrades install 'libnccl2=2.16.*+cuda11.8' 'libnccl-dev=2.16.*+cuda11.8' 'libcutensor1=1.6.*' 'libcutensor-dev=1.6.*' 'libcusparselt0=0.2.0.*' 'libcusparselt-dev=0.2.0.*' 'libcudnn8=8.7.*+cuda11.8' 'libcudnn8-dev=8.7.*+cuda11.8'

ENV PATH "/usr/lib/ccache:${PATH}"

COPY setup/update-alternatives-cutensor.sh /
RUN /update-alternatives-cutensor.sh

RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.11.0 && \
    pyenv global 3.11.0 && \
    pip install -U setuptools pip

RUN pip install -U 'numpy==1.23.*' 'scipy==1.9.*' 'optuna==3.*' 'mpi4py==3.*' 'cython==0.29.*'
RUN pip uninstall -y cuda-python && \
    pip check
