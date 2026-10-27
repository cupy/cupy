#!/bin/bash

set -uex

git clone --recurse-submodules https://github.com/data-apis/array-api-tests
pushd array-api-tests
python3 -m pip install --user -r requirements.txt
ARRAY_API_TESTS_MODULE=cupy pytest
popd
