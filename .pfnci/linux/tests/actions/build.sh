#!/bin/bash

set -uex

time python3 -m pip install --user -v ".[test]"
