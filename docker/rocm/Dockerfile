FROM rocm/dev-ubuntu-24.04:7.0.2
LABEL maintainer="CuPy Team"

RUN curl -qL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    hipblas hipsparse rocsparse hiprand rocrand rocthrust rocsolver rocfft hipfft hipcub rocprim rccl && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3.12 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN python3 -m pip install --no-cache-dir --break-system-packages -f https://pip.cupy.dev/pre/ "cupy-rocm-7-0[all]==14.0.0rc1"

ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
RUN python3 -c "import cupy; cupy.show_config()"
