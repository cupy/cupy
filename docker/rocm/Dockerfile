FROM rocm/dev-ubuntu-24.04:7.0.2
LABEL maintainer="CuPy Team"

RUN curl -qL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates hipblas hipsparse rocsparse hiprand rocrand rocthrust rocsolver rocfft hipfft hipcub rocprim rccl && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3.12 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Ubuntu's python3-pip (24.0, shipped by the base image) trusts the
# compromised eMudhra/emSign CAs, so replace it with pip from a hash-checked
# wheel run as its own installer. Keep the URL in sync with
# .github/actions/harden-pip/action.yml.
RUN apt-get purge -y python3-pip && \
    curl -fsSLo /tmp/pip-26.2.2-py3-none-any.whl https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl && \
    echo 'fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70  /tmp/pip-26.2.2-py3-none-any.whl' | sha256sum -c - && \
    python3 /tmp/pip-26.2.2-py3-none-any.whl/pip install --no-cache-dir --break-system-packages --disable-pip-version-check /tmp/pip-26.2.2-py3-none-any.whl && \
    rm /tmp/pip-26.2.2-py3-none-any.whl && \
    python3 -m pip install --no-cache-dir --break-system-packages --disable-pip-version-check --upgrade pip

RUN python3 -m pip install --no-cache-dir --break-system-packages -f https://pip.cupy.dev/pre/ "cupy-rocm-7-0[all]==14.0.0rc1"

ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
RUN python3 -c "import cupy; cupy.show_config()"
