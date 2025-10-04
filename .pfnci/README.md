# CuPy CI

CuPy uses two infrastructures for GPU tests.

* [PFN CI](https://ci.preferred.jp/) (`pfn-public-ci`): Tests for NVIDIA CUDA + Linux/Windows. Hosted on GCP.
* [Self-Hosted CI](https://github.com/cupy/self-hosted-ci/actions): Tests for AMD ROCm. GitHub Actions runners hosted on-premise.

Test matrices are defined by the following files:

* [`schema.yaml`](schema.yaml) defines all the possible values for each test axis (e.g., `cuda`, `python`, `numpy`), and constraints between them.
* [`matrix.yaml`](matrix.yaml) defines the configuration of each test matrix (e.g., `cupy.linux.cuda130`.)
* [`config.pbtxt`](config.pbtxt) defines the launch configuration (GPUs, memory, disk) of each test matrix.

The [`generate.py`](generate.py) tool takes these files as input and generates test assets (e.g., Dockerfiles/shell scripts for Linux) for each matrix, which are stored under [`linux/tests/`](linux/tests/) directory.
The tool also generates:

* [`config.tags.json`](config.tags.json): mapping of test matrices and test trigger phrases (e.g., matrices to be triggered by `/test mini`)
* [`coverage.rst`](coverage.rst): human-readable configuration coverage table

## Guide for Contributors

You can reproduce the CI environment using Docker.

```
# Build a Docker image for the `cupy.linux.cuda129.multi` test, then run the test for the current source tree within that image.
# Cache will be persisted at CACHE_DIR, which is recommended.
CACHE_DIR=/tmp/cupy-ci-cache-dir DOCKER_IMAGE_CACHE=0 ./.pfnci/linux/run.sh cuda129.multi build test
```

See [`linux/README.md`](linux/README.md) for the detailed usage.

## Guide for CI Maintainers

After modifying the test matrices, you need to regenerate the files by:

```
pip install PyYAML
./generate.py
```

### Future work

* Support generating Windows test environment.
* Support coverage reporting.

## Acknowledgement

CuPy's CI infrastructure is provided courtesy of [Preferred Networks](https://www.preferred.jp/en/).
