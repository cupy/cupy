FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Remove the pinned pip once Ubuntu's python3-pip is >= 26.2.2 (see .github/actions/harden-pip).
# noble's python3-pip is 24.0, which trusts only its vendored certifi.
# --ignore-installed: the distro pip has no RECORD file, so pip cannot
# uninstall it; install the new pip into /usr/local alongside it instead.
RUN python3 -m pip install --no-cache-dir --break-system-packages --disable-pip-version-check --ignore-installed \
        'pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70' && \
    python3 -m pip install --no-cache-dir --break-system-packages --disable-pip-version-check --upgrade pip

RUN pip3 install --no-cache-dir --break-system-packages -f https://pip.cupy.dev/pre/ "cupy-cuda13x[all]==14.0.0rc1"
