ARG BASE_IMAGE="rapidsai/base:25.02a-cuda12.5-py3.11"
FROM ${BASE_IMAGE}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install build-essential cuda-toolkit-12-5 ccache git curl

ENV PATH "/usr/lib/ccache:${PATH}"
ENV DISABLE_JUPYTER true
