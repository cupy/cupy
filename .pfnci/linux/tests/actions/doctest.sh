#!/bin/bash

set -uex

python3 -m pip install --user -r docs/requirements.txt

export SPHINXOPTS=-W

pushd docs
make html
make doctest
popd
