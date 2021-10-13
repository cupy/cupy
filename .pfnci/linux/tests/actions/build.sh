#!/bin/bash

set -uex

src_dir=.
if ! touch .; then
    src_dir=$(mktemp -d)
    echo "Source directory ($(pwd)) is read-only; copying the source tree to ${src_dir}"
    cp -a . "${src_dir}"
fi

pushd "${src_dir}"
time CUPY_NUM_NVCC_THREADS=8 CUPY_NUM_BUILD_JOBS=8 python3 -m pip install --user -v ".[test]"
popd
