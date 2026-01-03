#!/bin/bash

set -uex

# Use the "--no-build-isolation" option to test building with dependencies
# installed in the current environment. Otherwise builds always run in an
# isolated environment with the latest version of the build-system
# requirements specified in `pyproject.toml` and `setup_requires`.
time python3 -m pip install --no-build-isolation --user -v ".[test]"
