[![pypi](https://img.shields.io/pypi/dm/chainer.svg)](https://pypi.python.org/pypi/chainer)
[![pypi](https://img.shields.io/pypi/v/chainer.svg)](https://pypi.python.org/pypi/chainer)
[![GitHub license](https://img.shields.io/github/license/pfnet/chainer.svg)](https://github.com/pfnet/chainer)
[![travis](https://img.shields.io/travis/pfnet/chainer.svg)](https://travis-ci.org/pfnet/chainer)
[![coveralls](https://img.shields.io/coveralls/pfnet/chainer.svg)](https://coveralls.io/github/pfnet/chainer)

# Chainer: a neural network framework

## Requirements

Minimum requirements:
- Python 2.7+ or 3.4+
- NumPy 1.9+
- Six 1.9+

Requirements for some features:
- CUDA support
  - CUDA 6.5+
- CuDNN support
  - CuDNN v2
- Caffe model support
  - Python 2.7+ (Py3 is not supported)
  - Protocol Buffers (pip install protobuf)
- Testing utilities
  - Nose

## Installation

Install Chainer via PyPI:
```
pip install chainer
```

You can also install Chainer from the source code:
```
python setup.py install
```

If you want to enable CUDA, first you have to install CUDA and set the environment variable `PATH` and `LD_LIBRARY_PATH` for CUDA executables and libraries.
For example, if you are using Ubuntu and CUDA is installed by the official distribution, then CUDA is installed at `/usr/local/cuda`.
In this case, you have to add the following line to `.bashrc` or `.zshrc` (choose which you are using):
```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
Do not forget to restart your terminal session (or `source` it) to enable this change.
It automatically enables the CUDA support.

## More information

- Official site: http://chainer.org/
- Official document: http://docs.chainer.org/
- github: https://github.com/pfnet/chainer
- Forum: https://groups.google.com/forum/#!forum/chainer


## License

MIT License (see `LICENSE` file).
