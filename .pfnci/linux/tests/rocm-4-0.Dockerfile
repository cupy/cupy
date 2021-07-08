FROM rocm/dev-centos-7:4.0

RUN yum -y install rocm-dev hipblas hipfft hipsparse rocsparse rocrand rocthrust rocsolver rocfft hipcub rocprim rccl && \
    yum -y install ccache

ENV PATH="/usr/lib/ccache:${PATH}"
ENV ROCM_HOME="/opt/rocm"
ENV LD_LIBRARY_PATH="${ROCM_HOME}/lib"
ENV CPATH="${ROCM_HOME}/include"
ENV LDFLAGS="-L${ROCM_HOME}/lib"

# Python 3.6
RUN yum -y install python36-devel && \
    python3 -m pip install -U pip setuptools

RUN python3 -m pip install cython numpy scipy
