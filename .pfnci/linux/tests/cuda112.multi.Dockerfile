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
    yum -y install 'libnccl-2.16.*-*+cuda11.8' 'libnccl-devel-2.16.*-*+cuda11.8' 'libcusparselt0-0.2.0.*' 'libcusparselt-devel-0.2.0.*' 'libcudnn8-8.8.*-*.cuda11.8' 'libcudnn8-devel-8.8.*-*.cuda11.8'

ENV PATH "/usr/lib64/ccache:${PATH}"

RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.9.6 && \
    pyenv global 3.9.6 && \
    pip install -U setuptools pip wheel

RUN pip install -U 'numpy==1.22.*' 'scipy==1.7.*' 'optuna==3.*' 'cython==0.29.*'
RUN pip uninstall -y mpi4py cuda-python && \
    pip check
