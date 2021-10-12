ARG BASE_IMAGE="rapidsai/rapidsai-core-dev:21.10-cuda11.2-devel-ubuntu20.04-py3.8"
FROM ${BASE_IMAGE}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install ccache git curl

ENV PATH "/usr/lib/ccache:${PATH}"
