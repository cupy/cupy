#!/bin/bash

set -uex

src_dir=/src
if ! touch "${src_dir}"; then
    echo "Source directory (${src_dir}) is read-only; copying the source tree to /src-build"
    cp -a /src /src-build
    src_dir=/src-build
fi

pushd "${src_dir}"
time python3 -m pip install --user -v ".[test]"
popd
