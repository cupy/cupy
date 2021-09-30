FROM rocm/rocm-terminal:4.0.1
LABEL maintainer="CuPy Team"

USER root
RUN curl -qL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools \
    hipblas hipsparse rocsparse rocrand rocthrust rocsolver rocfft hipcub rocprim rccl && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir -U install setuptools pip
RUN pip3 install --no-cache-dir -f https://github.com/cupy/cupy/releases/v10.0.0b3 "cupy-rocm-4-0[all]==10.0.0b3"

USER rocm-user
