#!/bin/bash

set -uex

ACTIONS="$(dirname $0)/actions"
. "$ACTIONS/_environment.sh"

export NVCC="ccache nvcc"

"$ACTIONS/build.sh"

git clone https://github.com/data-apis/array-api-tests
pushd array-api-tests
python3 -m pip install --user -r requirements.txt
ARRAY_API_TESTS_MODULE=cupy pytest
popd

"$ACTIONS/cleanup.sh"
