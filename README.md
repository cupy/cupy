<div align="center"><img src="docs/image/cupy_logo_1000px.png" width="400"/></div>

# CuPy : NumPy-like API accelerated with CUDA

[![pypi](https://img.shields.io/pypi/v/cupy.svg)](https://pypi.python.org/pypi/cupy)
[![GitHub license](https://img.shields.io/github/license/cupy/cupy.svg)](https://github.com/cupy/cupy)
[![travis](https://img.shields.io/travis/cupy/cupy.svg)](https://travis-ci.org/cupy/cupy)
[![coveralls](https://img.shields.io/coveralls/cupy/cupy.svg)](https://coveralls.io/github/cupy/cupy)
[![Read the Docs](https://readthedocs.org/projects/cupy/badge/?version=stable)](http://docs.cupy.chainer.org/en/stable/?badge=stable)

[**Website**](https://cupy.chainer.org/)
| [**Docs**](http://docs.cupy.chainer.org/en/stable/)
| [**Install Guide**](http://docs.cupy.chainer.org/en/stable/install.html)
| [**Tutorial**](http://docs.cupy.chainer.org/en/stable/tutorial/)
| **Examples** ([Official](https://github.com/cupy/cupy/blob/master/examples))
| **Forum** ([en](https://groups.google.com/forum/#!forum/cupy), [ja](https://groups.google.com/forum/#!forum/cupy-jp))

*CuPy* is an implementation of NumPy-compatible multi-dimensional array on CUDA.
CuPy consists of the core multi-dimensional array class, `cupy.ndarray`, and many functions on it.
It supports a subset of `numpy.ndarray` interface.

## Requirements

CuPy is tested on Ubuntu 14.04 and CentOS 7. We recommend them to use CuPy, though it may run on other systems as well.

Minimum requirements:
- Python 2.7.6+, 3.4.3+, 3.5.1+, 3.6.0+
- NumPy 1.9, 1.10, 1.11, 1.12
- Six 1.9

Requirements for some features:
- CUDA support
  - CUDA 7.0, 7.5, 8.0
  - g++ 4.8.4+
- cuDNN support
  - cuDNN v4, v5, v5.1, v6
- Testing utilities
  - Mock
  - Nose

## Installation

### Minimum installation

If you use old ``setuptools``, upgrade it:

```
pip install -U setuptools
```

Then, install CuPy via PyPI:
```
pip install cupy
```

You can also install CuPy from the source code:
```
python setup.py install
```


### Installation with CUDA

If you want to enable CUDA, first you have to install CUDA and set the environment variable `PATH` and `LD_LIBRARY_PATH` for CUDA executables and libraries.
For example, if you are using Ubuntu and CUDA is installed by the official distribution, then CUDA is installed at `/usr/local/cuda`.
In this case, you have to add the following lines to `.bashrc` or `.zshrc` (choose which you are using):
```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```


If you want to enable cuDNN, add a directory containing `cudnn.h` to `CFLAGS`, and add a directory containing `libcudnn.so` to `LDFLAGS` and `LD_LIBRARY_PATH`:
```
export CFLAGS=-I/path/to/cudnn/include
export LDFLAGS=-L/path/to/cudnn/lib
export LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH
```
Do not forget to restart your terminal session (or `source` it) to enable these changes.
And then, reinstall CuPy.


### Multi-GPU Support

Multi-GPU training is supported by MultiprocessParallelUpdater.
If you want to use MultiprocessParallelUpdater, please install [NCCL](https://github.com/NVIDIA/nccl) by following the installation guide.


## Run with Docker

We provide the official Docker image.
Use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) command to run CuPy image with GPU.
You can login to the environment with bash, and run the Python interpreter.

```
$ nvidia-docker run -it cupy/cupy /bin/bash
```

## Development

Build CuPy from the source code as:

```
python setup.py develop
```

Run all tests as:

```
python -m unittest discover test "test_*.py"
```

Run a specific test file, or a specific test method respectively:

```
python -m unittest tests/cupy_tests/cuda_tests/test_memory.py
python -m unittest tests.cupy_tests.cuda_tests.test_memory.TestMemoryPointer.test_int
```

### Rebuild pxd files

Currently, we have problems that cython does not rebuild pxd files well with `python setup.py develop`.

Clean `*.cpp` and `*.so` files once with:

```
git clean -fdx
```

Then, run `python setup.py develop` again.

### ccache

We do not officially support, but some of the developer members use [ccache](https://ccache.samba.org/) to boost compilation time.

For example, on Ubuntu, set up as followings:

```
sudo apt-get install ccache
export PATH=/usr/lib/ccache:$PATH
```

See [ccache](https://ccache.samba.org/) for details.

## More information

- [Release notes](https://github.com/cupy/cupy/releases)
- [Projects using CuPy](https://github.com/cupy/cupy/wiki/Projects-using-CuPy)

## License

MIT License (see `LICENSE` file).
