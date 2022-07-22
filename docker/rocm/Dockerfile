FROM rocm/rocm-terminal:5.0.1
LABEL maintainer="CuPy Team"

USER root
RUN curl -qL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    hipblas hipsparse rocsparse rocrand rocthrust rocsolver rocfft hipcub rocprim rccl && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends python3.7 python3.7-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1

RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.7

RUN pip3 install --no-cache-dir -U install setuptools pip
RUN pip3 install --no-cache-dir -f https://github.com/cupy/cupy/releases/v11.0.0rc1 "cupy-rocm-5-0[all]==11.0.0rc1"

USER rocm-user
