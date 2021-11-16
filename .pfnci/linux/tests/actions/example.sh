#!/bin/bash

set -uex

###
### Examples
###

# TODO: support coverage reporting
python3 -m pip install --user matplotlib

export MPLBACKEND=Agg

pushd examples

# K-means
python3 kmeans/kmeans.py -m 1
python3 kmeans/kmeans.py -m 1 --use-custom-kernel
python3 kmeans/kmeans.py -m 1 -o kmeans.png

# SGEMM
PYTHONPATH=gemm python3 gemm/sgemm.py

# cg
python3 cg/cg.py

# cuTENSOR
python3 cutensor/contraction.py
python3 cutensor/elementwise_binary.py
python3 cutensor/elementwise_trinary.py
python3 cutensor/reduction.py

# stream
python3 stream/cublas.py
python3 stream/cudnn.py
python3 stream/cufft.py
python3 stream/cupy_event.py
python3 stream/cupy_kernel.py
python3 stream/cupy_memcpy.py
python3 stream/curand.py
python3 stream/cusolver.py
python3 stream/cusparse.py
python3 stream/map_reduce.py
python3 stream/thrust.py

popd


###
### Doctest
###

pushd docs
python3 -m pip install --user -r requirements.txt
SPHINXOPTS=-W make doctest
popd
