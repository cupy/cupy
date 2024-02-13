ARG BASE_IMAGE="rapidsai/base:23.12-cuda12.0-py3.10"
FROM ${BASE_IMAGE}

USER root

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install build-essential cuda-toolkit-12-0 ccache git curl

ENV CUDA_PATH "/usr/local/cuda"
ENV PATH "/usr/lib/ccache:${PATH}"
