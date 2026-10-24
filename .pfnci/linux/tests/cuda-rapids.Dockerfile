ARG BASE_IMAGE="rapidsai/base:25.02-cuda12.8-py3.11"
FROM ${BASE_IMAGE}

USER root

# Install pip 26.2.2 from a hash-pinned URL before any other use of pip (the
# eMudhra/emSign CA incident, https://discuss.python.org/t/109576). Keep in
# sync with PIP_SPEC in generate.py.
RUN python -m pip install --disable-pip-version-check 'pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70'

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install build-essential cuda-toolkit-12-8 ccache git curl

ENV CUDA_PATH "/usr/local/cuda"
ENV PATH "/usr/lib/ccache:${PATH}"
