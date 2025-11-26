# CuPy CI

CuPy uses the following two infrastructures for GPU tests.

* [PFN CI](https://ci.preferred.jp/) (`pfn-public-ci`): Tests for NVIDIA CUDA + Linux/Windows. Hosted on GCP.
* [Self-Hosted CI](https://github.com/cupy/self-hosted-ci/actions): Tests for AMD ROCm. GitHub Actions runners hosted on-premise.

Test matrices are defined by the following files:

* [`schema.yaml`](schema.yaml) defines all the possible values for each test axis (e.g., `cuda`, `python`, `numpy`), and constraints between them.
* [`matrix.yaml`](matrix.yaml) defines the configuration of each test matrix (e.g., `cupy.linux.cuda130`.)
* [`config.pbtxt`](config.pbtxt) (for PFN CI) defines the launch configuration (GPUs, memory, disk) of each test matrix.
* [`self-hosted-ci.yml`](../.github/workflows/self-hosted-ci.yml) workflow (for Self-Hosted CI) defines the list of test matrices to trigger. The workflow triggers the actual [CI workflow](https://github.com/cupy/self-hosted-ci/blob/main/.github/workflows/ci.yml) in a separate repository dedicated for self-hosted runners for security reasons.

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

### Timeouts

CuPy CI has a two layers of timeouts:

* Timeout set by CI system
  * Configured by `time_limit` in `config.pbtxt` (for PFN CI) and `timeout-minutes` in workflow files (for GitHub Actions workflows including self-hosted CIs).
* Timeout set by CI scripts
  * Configured by `timeout` command in [Linux](https://github.com/cupy/cupy/blob/main/.pfnci/linux/tests/actions/unittest.sh) and  `RunWithTimeout` in [Windows](https://github.com/cupy/cupy/blob/main/.pfnci/windows/test.ps1).

The CI system-level timeout is a hard limit designed to prevent situations where instances fail to terminate due to unexpected hangs. On the other hand, the CI script-level timeout is set to ensure that whole CI processes complete gracefully without exceeding the system-level timeout. This allows preserving sufficient time for finalization actions like uploading cache, even if the unit test execution itself times out.

### Future work

* Support generating Windows test environment.
* Support coverage reporting.

## Acknowledgement

CuPy's CI infrastructure is provided courtesy of [Preferred Networks](https://www.preferred.jp/en/).
