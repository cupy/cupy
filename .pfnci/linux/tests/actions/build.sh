#!/bin/bash

set -uex

src_dir=.
if ! touch .; then
    src_dir=$(mktemp -d)
    echo "Source directory ($(pwd)) is read-only; copying the source tree to ${src_dir}"
    cp -a . "${src_dir}"
fi

pushd "${src_dir}"
time python3 -m pip install --user -v ".[test]"
popd
