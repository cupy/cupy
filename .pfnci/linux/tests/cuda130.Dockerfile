# AUTO GENERATED: DO NOT EDIT!
ARG BASE_IMAGE="nvidia/cuda:13.0.0-devel-ubuntu24.04"
FROM ${BASE_IMAGE}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install \
       make build-essential libssl-dev zlib1g-dev \
       libbz2-dev libreadline-dev libsqlite3-dev wget \
       curl llvm libncursesw5-dev xz-utils tk-dev \
       libxml2-dev libxmlsec1-dev libffi-dev \
       liblzma-dev \
\
       && \
    apt-get -qqy install ccache git curl && \
    apt-get -qqy --allow-change-held-packages \
            --allow-downgrades install 'libnccl2=2.27.*+cuda13.0' 'libnccl-dev=2.27.*+cuda13.0' 'libcusparselt0=0.7.1.*' 'libcusparselt-dev=0.7.1.*'

ENV PATH "/usr/lib/ccache:${PATH}"

RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.12.6 && \
    pyenv global 3.12.6 && \
    pip install -U setuptools pip wheel

RUN pip install -U 'numpy==2.3.*' 'scipy==1.16.*' 'optuna==4.*' 'cython==3.1.*' 'fastrlock>=0.5'
RUN pip uninstall -y mpi4py cuda-python && \
    pip check
