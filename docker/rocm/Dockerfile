FROM rocm/rocm-terminal:5.0.1
LABEL maintainer="CuPy Team"

USER root
RUN curl -qL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    hipblas hipsparse rocsparse rocrand rocthrust rocsolver rocfft hipcub rocprim rccl && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends python3.8 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1

RUN python3 -m pip install --no-cache-dir -U install setuptools pip
RUN python3 -m pip install --no-cache-dir -f https://pip.cupy.dev/pre/ "cupy-rocm-5-0[all]==13.0.0a1"

USER rocm-user
