ARG BASE_IMAGE="rapidsai/rapidsai-core-dev-nightly:22.08-cuda11.5-devel-ubuntu20.04-py3.9"
FROM ${BASE_IMAGE}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install ccache git curl

ENV PATH "/usr/lib/ccache:${PATH}"
