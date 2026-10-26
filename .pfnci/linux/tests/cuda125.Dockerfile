# AUTO GENERATED: DO NOT EDIT!
ARG BASE_IMAGE="nvidia/cuda:12.5.0-devel-ubuntu22.04"
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
    apt-get -qqy install ca-certificates ccache git curl && \
    apt-get -qqy --allow-change-held-packages \
            --allow-downgrades install 'libnccl2=2.21.*+cuda12.5' 'libnccl-dev=2.21.*+cuda12.5' 'libcutensor2-cuda-12=2.4.*' 'libcutensor2-dev-cuda-12=2.4.*'

ENV PATH "/usr/lib/ccache:${PATH}"

RUN curl -fsSL https://github.com/cli/cli/releases/download/v2.95.0/gh_2.95.0_linux_amd64.tar.gz \
        | tar -xz -C /usr/local --strip-components=1 gh_2.95.0_linux_amd64/bin/gh

ENV CUPY_INCLUDE_PATH=/usr/include/libcutensor/12:${CUPY_INCLUDE_PATH}
ENV CUPY_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12:${CUPY_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcutensor/12:${LD_LIBRARY_PATH}
RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN git -C /opt/pyenv pull --ff-only && \
    PYTHON_CONFIGURE_OPTS="--disable-shared" pyenv install 3.12.11 && \
    pyenv global 3.12.11 && \
    # Remove the pinned pip once every Python in schema.yaml ships pip>=26.2.2 or blocks the eMudhra CAs.
    pip install 'pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70' && \
    pip install -U pip setuptools wheel && \
    pip install -U google-cloud-storage

RUN pip install -U 'numpy==2.0.*' 'scipy==1.15.*' 'optuna==3.*' 'ml_dtypes==0.5.*' 'cython==3.2.*,!=3.2.6'
RUN pip uninstall -y mpi4py cuda-python nvmath-python cuda-cccl && \
    pip check

RUN mkdir /home/cupy-user && chmod 777 /home/cupy-user
