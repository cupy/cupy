# AUTO GENERATED: DO NOT EDIT!
ARG BASE_IMAGE="nvidia/cuda:11.2.1-devel-centos8"
FROM ${BASE_IMAGE}

RUN yum -y install \
       zlib-devel bzip2 bzip2-devel readline-devel sqlite \
       sqlite-devel openssl-devel tk-devel libffi-devel \
       xz-devel && \
    yum -y install epel-release && \
    yum -y install "@Development Tools" ccache git curl && \
    yum -y install libnccl-devel-2.8.*-*+cuda11.2 libcutensor-devel-1.3.* libcusparselt-devel-0.2.0.* libcudnn8-devel-8.1.*-*.cuda11.2

ENV PATH "/usr/lib64/ccache:${PATH}"

RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.7.11 && \
    pyenv global 3.7.11 && \
    pip install -U setuptools pip

RUN pip install -U numpy==1.18.* scipy==1.5.* optuna==2.* cython==0.29.*
