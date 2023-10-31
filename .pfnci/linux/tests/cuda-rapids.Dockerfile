ARG BASE_IMAGE="rapidsai/base:23.10-cuda11.8-py3.10"
FROM ${BASE_IMAGE}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install ccache git curl

ENV PATH "/usr/lib/ccache:${PATH}"
ENV DISABLE_JUPYTER true
