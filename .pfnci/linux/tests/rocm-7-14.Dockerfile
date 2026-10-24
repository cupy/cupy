# AUTO GENERATED: DO NOT EDIT!
ARG BASE_IMAGE="rocm/dev-ubuntu-22.04:7.14.0-full"
FROM ${BASE_IMAGE}

RUN export DEBIAN_FRONTEND=noninteractive && \
    ( apt-get -qqy update || true ) && \
    apt-get -qqy install ca-certificates gnupg && \
    curl -qL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
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
            --allow-downgrades install 

ENV PATH "/usr/lib/ccache:${PATH}"

RUN curl -fsSL https://github.com/cli/cli/releases/download/v2.95.0/gh_2.95.0_linux_amd64.tar.gz \
        | tar -xz -C /usr/local --strip-components=1 gh_2.95.0_linux_amd64/bin/gh

ENV ROCM_HOME "/opt/rocm"
ENV LD_LIBRARY_PATH "${ROCM_HOME}/lib"
ENV CPATH "${ROCM_HOME}/include"
ENV LDFLAGS "-L${ROCM_HOME}/lib"

RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN git -C /opt/pyenv pull --ff-only && \
    PYTHON_CONFIGURE_OPTS="--disable-shared" pyenv install 3.11.13 && \
    pyenv global 3.11.13 && \
    # Remove the pinned pip once every Python in schema.yaml ships pip>=26.2.2 or blocks the eMudhra CAs.
    pip install 'pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70' && \
    pip install -U pip setuptools wheel && \
    pip install -U google-cloud-storage

RUN pip install -U 'numpy==2.3.*' 'scipy==1.16.*' 'optuna==3.*' 'ml_dtypes==0.5.*' 'cython==3.2.*,!=3.2.6'
RUN pip uninstall -y mpi4py cuda-python nvmath-python cuda-cccl && \
    pip check

RUN mkdir /home/cupy-user && chmod 777 /home/cupy-user
