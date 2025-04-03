# AUTO GENERATED: DO NOT EDIT!
ARG BASE_IMAGE="nvidia/cuda:11.2.2-devel-centos7"
FROM ${BASE_IMAGE}

COPY setup/setup-yum-centos7-pre.sh setup/setup-yum-centos7-post.sh /

RUN /setup-yum-centos7-pre.sh && \
    yum -y install centos-release-scl && \
    /setup-yum-centos7-post.sh && \
    yum -y install devtoolset-7-gcc-c++
ENV PATH "/opt/rh/devtoolset-7/root/usr/bin:${PATH}"
ENV LD_LIBRARY_PATH "/opt/rh/devtoolset-7/root/usr/lib64:/opt/rh/devtoolset-7/root/usr/lib:${LD_LIBRARY_PATH}"

RUN yum -y install \
       zlib-devel bzip2 bzip2-devel readline-devel sqlite \
       sqlite-devel openssl-devel tk-devel libffi-devel \
       xz-devel && \
    yum -y install epel-release && \
    yum -y install "@Development Tools" ccache git curl && \
    yum -y install 'libnccl-2.16.*-*+cuda11.8' 'libnccl-devel-2.16.*-*+cuda11.8' 'libcudnn8-8.8.*-*.cuda11.8' 'libcudnn8-devel-8.8.*-*.cuda11.8'

ENV PATH "/usr/lib64/ccache:${PATH}"

RUN yum -y install openssl11-devel
ENV CFLAGS "-I/usr/include/openssl11"
ENV CPPFLAGS "-I/usr/include/openssl11"
ENV LDFLAGS "-L/usr/lib64/openssl11"

RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.10.15 && \
    pyenv global 3.10.15 && \
    pip install -U setuptools pip wheel

RUN pip install -U 'numpy==1.24.*' 'scipy==1.10.*' 'optuna==3.*' 'cython==3.*'
RUN pip uninstall -y mpi4py cuda-python && \
    pip check
