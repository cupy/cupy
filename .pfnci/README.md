# CuPy CI

CuPy uses two infrastructures for GPU tests.

* FlexCI (`pfn-public-ci`): GCP, Linux/Windows + CUDA tests
* Jenkins: on-premise, Linux + ROCm tests

This directory contains the test matrix definition, and a tool to generate test environment from the matrix.

* [`schema.yaml`](schema.yaml) defines all the possible values for each test axis (e.g., `cuda`, `python`, `numpy`), and constraints between them.
* [`matrix.yaml`](matrix.yaml) defines the configuration of each test matrix (e.g., `cupy.linux.cuda115`.)
* [`generate.py`](generate.py) generates the test assets (Dockerfile/shell script for Linux, PowerShell script for Windows) for each matrix from the schema and the matrix.
  This program also generates the following files:
    * [`config.tags.json`](config.tags.json): mapping of test matrices and test trigger phrases
    * [`coverage.rst`](coverage.rst): human-readable configuration coverage table
* [`config.pbtxt`](config.pbtxt) is a FlexCI configuration file that defines launch configurations of each test matrix.

## Usage

To generate `linux/tests/*.Dockerfile`, `linux/tests/*.sh`, `config.tags.json`, and `coverage.rst`:

```
pip install PyYAML
./generate.py
```

## Future work

* Support generating Windows test environment.
* Generate shuffle tests from `schema.yaml`.
* Support using OS-provided Python binary packages instead of pyenv.
* Support coverage reporting.
* Support installation tests.
