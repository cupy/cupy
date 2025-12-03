# AUTO GENERATED: DO NOT EDIT!
ARG BASE_IMAGE="nvidia/cuda:13.0.0-devel-ubuntu22.04"
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
            --allow-downgrades install 'libnccl2=2.27.*+cuda13.0' 'libnccl-dev=2.27.*+cuda13.0' 'libcutensor2-cuda-13=2.3.*' 'libcutensor2-dev-cuda-13=2.3.*' 'libcusparselt0-cuda-13=0.8.1.*' 'libcusparselt0-dev-cuda-13=0.8.1.*'

ENV PATH "/usr/lib/ccache:${PATH}"

ENV CUPY_INCLUDE_PATH=/usr/include/libcutensor/13:${CUPY_INCLUDE_PATH}
ENV CUPY_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/13:${CUPY_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/13:${LD_LIBRARY_PATH}
RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.14.0 && \
    pyenv global 3.14.0 && \
    pip install -U setuptools pip wheel

RUN pip install -U 'numpy==2.3.*' 'scipy==1.16.*' 'optuna==4.*' 'cython==3.1.*'
RUN pip uninstall -y mpi4py cuda-python && \
    pip check

RUN mkdir /home/cupy-user && chmod 777 /home/cupy-user
