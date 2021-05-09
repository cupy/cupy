FROM rocm/rocm-terminal:4.0.1
LABEL maintainer="CuPy Team"

USER root
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    hipblas hipsparse rocsparse rocrand rocthrust rocsolver rocfft hipcub rocprim rccl && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir -U install setuptools pip
RUN pip3 install --no-cache-dir "cupy-rocm-4-0[all]==9.0.0rc1"

USER rocm-user
