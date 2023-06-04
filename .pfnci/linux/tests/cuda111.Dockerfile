# AUTO GENERATED: DO NOT EDIT!
ARG BASE_IMAGE="nvidia/cuda:11.1.1-devel-centos7"
FROM ${BASE_IMAGE}

RUN yum -y install centos-release-scl && \
    yum -y install devtoolset-7-gcc-c++
ENV PATH "/opt/rh/devtoolset-7/root/usr/bin:${PATH}"
ENV LD_LIBRARY_PATH "/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:${LD_LIBRARY_PATH}"

RUN yum -y install \
       zlib-devel bzip2 bzip2-devel readline-devel sqlite \
       sqlite-devel openssl-devel tk-devel libffi-devel \
       xz-devel && \
    yum -y install epel-release && \
    yum -y install "@Development Tools" ccache git curl && \
    yum -y install 'libnccl-2.8.*-*+cuda11.1' 'libnccl-devel-2.8.*-*+cuda11.1' 'libcutensor1-1.5.*' 'libcutensor-devel-1.5.*' 'libcudnn8-8.0.*-*.cuda11.1' 'libcudnn8-devel-8.0.*-*.cuda11.1'

ENV PATH "/usr/lib64/ccache:${PATH}"

COPY setup/update-alternatives-cutensor.sh /
RUN /update-alternatives-cutensor.sh

RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.8.11 && \
    pyenv global 3.8.11 && \
    pip install -U setuptools pip wheel

RUN pip install -U 'numpy==1.20.*' 'cython==0.29.*'
RUN pip uninstall -y scipy optuna mpi4py cuda-python && \
    pip check
