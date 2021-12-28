#!/bin/bash

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

export NVCC="ccache nvcc"

"$ACTIONS/build.sh"

git clone https://github.com/data-apis/array-api-tests
pushd array-api-tests
pip install -r requirements.txt
ARRAY_API_TESTS_MODULE=cupy.array_api pytest
popd

"$ACTIONS/cleanup.sh"
