FROM rocm/dev-ubuntu-24.04:7.0.2
LABEL maintainer="CuPy Team"

RUN curl -qL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates hipblas hipsparse rocsparse hiprand rocrand rocthrust rocsolver rocfft hipfft hipcub rocprim rccl && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3.12 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Remove the pinned pip once Ubuntu's python3-pip is >= 26.2.2 (see PIP_SPEC in .pfnci/generate.py).
# noble's python3-pip is 24.0, which trusts only its vendored certifi.
RUN python3 -m pip install --no-cache-dir --break-system-packages --disable-pip-version-check \
        'pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70' && \
    python3 -m pip install --no-cache-dir --break-system-packages --disable-pip-version-check --upgrade pip

RUN python3 -m pip install --no-cache-dir --break-system-packages -f https://pip.cupy.dev/pre/ "cupy-rocm-7-0[all]==14.0.0rc1"

ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
RUN python3 -c "import cupy; cupy.show_config()"
