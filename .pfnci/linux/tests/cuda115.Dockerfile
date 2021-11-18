###
### https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.5.0/ubuntu2004/base/Dockerfile
###
FROM ubuntu:20.04 as base

ENV NVARCH x86_64
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.5 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450"
ENV NV_CUDA_CUDART_VERSION 11.5.50-1
ENV NV_CUDA_COMPAT_PACKAGE cuda-compat-11-5

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH}/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    if [ ! -z ${NV_ML_REPO_ENABLED} ]; then echo "deb ${NV_ML_REPO_URL} /" > /etc/apt/sources.list.d/nvidia-ml.list; fi && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 11.5.0

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-5=${NV_CUDA_CUDART_VERSION} \
    ${NV_CUDA_COMPAT_PACKAGE} \
    && ln -s cuda-11.5 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility


###
### https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.5.0/ubuntu2004/devel/Dockerfile
###

ENV NV_CUDA_LIB_VERSION "11.5.0-1"

ENV NV_CUDA_CUDART_DEV_VERSION 11.5.50-1
ENV NV_NVML_DEV_VERSION 11.5.50-1
ENV NV_LIBCUSPARSE_DEV_VERSION 11.7.0.31-1
ENV NV_LIBNPP_DEV_VERSION 11.5.1.53-1
ENV NV_LIBNPP_DEV_PACKAGE libnpp-dev-11-5=${NV_LIBNPP_DEV_VERSION}

ENV NV_LIBCUBLAS_DEV_VERSION 11.7.3.1-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME libcublas-dev-11-5
ENV NV_LIBCUBLAS_DEV_PACKAGE ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtinfo5 libncursesw5 \
    cuda-cudart-dev-11-5=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-11-5=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-11-5=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-11-5=${NV_CUDA_LIB_VERSION} \
    cuda-nvml-dev-11-5=${NV_NVML_DEV_VERSION} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    libcusparse-dev-11-5=${NV_LIBCUSPARSE_DEV_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs


###
### CuPy
###

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install \
       make build-essential libssl-dev zlib1g-dev \
       libbz2-dev libreadline-dev libsqlite3-dev wget \
       curl llvm libncursesw5-dev xz-utils tk-dev \
       libxml2-dev libxmlsec1-dev libffi-dev \
       liblzma-dev && \
    apt-get -qqy install ccache git curl && \
    apt-get -qqy --allow-change-held-packages \
            --allow-downgrades install libnccl2=2.11.*+cuda11.5 libnccl-dev=2.11.*+cuda11.5 libcutensor-dev=1.3.* libcusparselt-dev=0.2.0.* libcudnn8=8.2.*+cuda11.4 libcudnn8-dev=8.2.*+cuda11.4

ENV PATH "/usr/lib/ccache:${PATH}"

RUN git clone https://github.com/pyenv/pyenv.git /opt/pyenv
ENV PYENV_ROOT "/opt/pyenv"
ENV PATH "${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN pyenv install 3.9.6 && \
    pyenv global 3.9.6 && \
    pip install -U setuptools pip

RUN pip install -U numpy==1.21.* scipy==1.7.* optuna==2.* cython==0.29.*
