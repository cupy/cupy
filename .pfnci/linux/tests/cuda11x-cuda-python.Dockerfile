FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy update && \
    apt-get -qqy install ccache && \
    apt-get -qqy --allow-change-held-packages \
            --allow-downgrades install libnccl2=2.11.*+cuda11.4 libnccl-dev=2.11.*+cuda11.4 libcutensor1=1.3.* libcutensor-dev=1.3.* libcusparselt0=0.1.0.* libcusparselt-dev=0.1.0.* libcudnn8=8.2.*+cuda11.4 libcudnn8-dev=8.2.*+cuda11.4

ENV PATH="/usr/lib/ccache:${PATH}"

# Python 3.8 (Ubuntu 20.04)
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qqy install python3 python3-dev python3-pip python3-setuptools && \
    python3 -m pip install -U pip setuptools && \
    apt-get -qqy purge python3-pip python3-setuptools
