FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python3-dev \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Ubuntu's python3-pip (24.0) trusts the compromised eMudhra/emSign CAs, so
# pip comes from a hash-checked wheel run as its own installer instead. Keep
# the URL in sync with docker/rocm/Dockerfile,
# .github/workflows/scripts/pretest-rocm-test.sh and .pfnci/windows/test.ps1.
RUN curl -fsSLo /tmp/pip-26.2.2-py3-none-any.whl https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl && \
    echo 'fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70  /tmp/pip-26.2.2-py3-none-any.whl' | sha256sum -c - && \
    python3 /tmp/pip-26.2.2-py3-none-any.whl/pip install --no-cache-dir --break-system-packages --disable-pip-version-check /tmp/pip-26.2.2-py3-none-any.whl && \
    rm /tmp/pip-26.2.2-py3-none-any.whl && \
    python3 -m pip install --no-cache-dir --break-system-packages --disable-pip-version-check --upgrade pip

RUN pip3 install --no-cache-dir --break-system-packages -f https://pip.cupy.dev/pre/ "cupy-cuda13x[all]==14.0.0rc1"
