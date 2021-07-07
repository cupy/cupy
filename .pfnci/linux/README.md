# Linux CI Scripts

This directory contains assets used for CI.
Currently, only ROCm and CUDA 11.4 tests are defined (many other tests are still using [chainer-test](http://github.com/chainer/chainer-test)), but we will gradually move all test variants to here.

All tests here are isolated by Docker so that developers and contributors can reproduce the same environment as the CI.
You can use the `run.sh` tool to build the Docker image and run unit tests in the image.
The current (local) codebase is read-only mounted to the container and used for testing.

`./run.sh` takes a TARGET and one or more STAGEs as arguments.
Here are some examples:

```
# Target: cuda114
# Stages: Build the docker image for testing, then run the unit test.
./run.sh cuda114 build test

# Target: rocm-4-0
# Stages: Only build the docker image.
./run.sh rocm-4-0 build
```

`tests/` directory contains Dockerfiles and bootstrap shell scripts with TARGET name prefixed.
For example, when the target is `rocm-4-2`, `rocm-4-2.Dockerfile` and `rocm-4-2.sh` are used for testing.

Run `./run.sh` without arguments for the detailed usage.
