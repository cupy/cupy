ARG BASE_IMAGE="rapidsai/rapidsai-core-dev:22.10-cuda11.5-devel-ubuntu20.04-py3.9"
FROM ${BASE_IMAGE}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install ccache git curl

ENV PATH "/usr/lib/ccache:${PATH}"
ENV DISABLE_JUPYTER true
