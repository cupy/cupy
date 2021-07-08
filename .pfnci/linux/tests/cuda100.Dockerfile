# AUTO GENERATED: DO NOT EDIT!
ARG BASE_IMAGE="nvidia/cuda:10.0-devel-centos7"
FROM ${BASE_IMAGE}

RUN yum -y install \
       zlib-devel bzip2 bzip2-devel readline-devel sqlite \
       sqlite-devel openssl-devel tk-devel libffi-devel \
       xz-devel && \
    yum -y install "@Development Tools" ccache git curl && \
    yum -y install libnccl-devel-2.6.*+cuda10.0 libcudnn7-devel-7.6.*+cuda10.0

ENV PATH "/usr/lib/ccache:${PATH}"

RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.6.14 && \
    pyenv global 3.6.14 && \
    pip install -U setuptools pip

RUN pip install -U numpy==1.17.* scipy==1.4.* optuna==2.* cython==0.29.*
