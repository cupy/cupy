# CuPy CI

CuPy uses two infrastructures for GPU tests.

* FlexCI (`pfn-public-ci`): GCP, Linux/Windows, CUDA only
* Jenkins: on-premise, Linux only, CUDA/ROCm

Currently most of test configurations are managed by [chainer-test](http://github.com/chainer/chainer-test), which contains a set of scripts for Jenkins, but we are gradually migrating to new tooling in this directory.
We are also gradually migrating from Jenkins to FlexCI for better performance; eventually Jenkins will only be used for ROCm tests.

This directory contains the test matrix definition, and a tool to generate test environment from the matrix.

* `schema.yaml` defines all the possible values for each test axis, and constraints between them.
* `matrix.yaml` defines the configuration of each matrix.
* `generate.py` generates the test environment (Dockerfile/shell script for Linux, PowerShell script for Windows) for each matrix from the schema and the matrix.
  This program also generates `coverage.md` to see the configuration coverage.

## Usage

To generate `linux/tests/*.Dockerfile`, `linux/tests/*.sh` and `coverage.md`:

```
pip install PyYAML
./generate.py -s schema.yaml -m matrix.yaml
```

## Future work

* Support generating Windows test environment.
* Test notifications to Gitter.
* Generate shuffle tests from `schema.yaml`.
* Support using OS-provided Python binary packages instead of pyenv.
