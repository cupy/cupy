ARG BASE_IMAGE="rapidsai/rapidsai-core-nightly:22.06-cuda11.2-runtime-ubuntu20.04-py3.8"
FROM ${BASE_IMAGE}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install ccache git curl

ENV PATH "/usr/lib/ccache:${PATH}"
