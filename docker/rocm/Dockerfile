FROM rocm/dev-ubuntu-20.04:5.0.1
LABEL maintainer="CuPy Team"

RUN curl -qL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    hipblas hipsparse rocsparse rocrand rocthrust rocsolver rocfft hipfft hipcub rocprim rccl && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3.9 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

RUN python3 -m pip install --no-cache-dir -U install setuptools pip
RUN python3 -m pip install --no-cache-dir -f https://pip.cupy.dev/pre/ "cupy-rocm-5-0[all]==13.0.0rc1"

ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
RUN python3 -c "import cupy; cupy.show_config()"
