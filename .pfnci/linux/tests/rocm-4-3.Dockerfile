FROM rocm/dev-ubuntu-20.04:4.3

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install rocm-dev hipblas hipfft hipsparse rocsparse rocrand rocthrust rocsolver rocfft hipcub rocprim rccl && \
    apt-get -qqy install ccache

ENV PATH="/usr/lib/ccache:${PATH}"
ENV ROCM_HOME="/opt/rocm"
ENV LD_LIBRARY_PATH="${ROCM_HOME}/lib"
ENV CPATH="${ROCM_HOME}/include"
ENV LDFLAGS="-L${ROCM_HOME}/lib"

# Python 3.8
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy install python3 python3-dev python3-pip python3-setuptools && \
    python3 -m pip install -U pip setuptools && \
    apt-get -qqy purge python3-pip python3-setuptools

RUN python3 -m pip install cython numpy scipy
