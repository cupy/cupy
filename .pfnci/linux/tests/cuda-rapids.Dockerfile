ARG BASE_IMAGE="rapidsai/base:26.10-cuda12-py3.11"
FROM ${BASE_IMAGE}

USER root

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install build-essential cuda-toolkit-12-9 ccache git curl

ENV CUDA_PATH "/usr/local/cuda"
ENV PATH "/usr/lib/ccache:${PATH}"
