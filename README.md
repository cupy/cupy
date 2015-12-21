[![pypi](https://img.shields.io/pypi/dm/chainer.svg)](https://pypi.python.org/pypi/chainer)
[![pypi](https://img.shields.io/pypi/v/chainer.svg)](https://pypi.python.org/pypi/chainer)
[![GitHub license](https://img.shields.io/github/license/pfnet/chainer.svg)](https://github.com/pfnet/chainer)
[![travis](https://img.shields.io/travis/pfnet/chainer.svg)](https://travis-ci.org/pfnet/chainer)
[![coveralls](https://img.shields.io/coveralls/pfnet/chainer.svg)](https://coveralls.io/github/pfnet/chainer)

# Chainer: a neural network framework

## Requirements

Chainer is tested on Ubuntu 14.04 and CentOS 7. We recommend them to use Chainer, though it may run on other systems as well.

Minimum requirements:
- Python 2.7.6+, 3.4.3+, 3.5.0+
- NumPy 1.9
- Six 1.9
- h5py 2.5.0

Requirements for some features:
- CUDA support
  - CUDA 6.5, 7.0, 7.5
  - filelock
  - g++
- cuDNN support
  - cuDNN v2, v3
- Caffe model support
  - Python 2.7.6+ (Py3 is not supported)
  - Protocol Buffers (pip install protobuf)
- Testing utilities
  - Mock
  - Nose

## Installation

Chainer requires libhdf5 via h5py. Anaconda distribution includes this package. If you are using another Python distribution, use either of the following commands to install libhdf5 depending on your Linux environment:

```
apt-get install libhdf5-dev
yum install hdf5-devel
```

If you use old ``setuptools``, upgrade it:

```
pip install -U setuptools
```

Then, install Chainer via PyPI:
```
pip install chainer
```

You can also install Chainer from the source code:
```
python setup.py install
```

If you want to enable CUDA, first you have to install CUDA and set the environment variable `PATH` and `LD_LIBRARY_PATH` for CUDA executables and libraries.
For example, if you are using Ubuntu and CUDA is installed by the official distribution, then CUDA is installed at `/usr/local/cuda`.
In this case, you have to add the following lines to `.bashrc` or `.zshrc` (choose which you are using):
```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Chainer had `chainer-cuda-deps` module to enable CUDA in previous version.
Recent version (>=1.3) does not require this module.
So **you do not have to install** `chainer-cuda-deps`.

If you want to enable cuDNN, add a directory containing `cudnn.h` to `CPATH`, and add a directory containing `libcudnn.so` to `LIBRARY_PATH` and `LD_LIBRARY_PATH`:
```
export CPATH=/path/to/cudnn/include:$CPATH
export LIBRARY_PATH=/path/to/cudnn/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/cudnn/lib:$LD_LIBRARY_PATH
```
Do not forget to restart your terminal session (or `source` it) to enable these changes.
And then, reinstall Chainer.


## Reference

Seiya Tokui, Kenta Oono, Shohei Hido and Justin Clayton,
*Chainer: a Next-Generation Open Source Framework for Deep Learning*,
in Neural Information Processing Systems(NIPS), Workshop on Machine Learning Systems(LearningSys), 2015
[URL](http://learningsys.org/papers/LearningSys_2015_paper_33.pdf), [BibTex](chainer_bibtex.txt)


## More information

- Official site: http://chainer.org/
- Official document: http://docs.chainer.org/
- github: https://github.com/pfnet/chainer
- Forum: https://groups.google.com/forum/#!forum/chainer

## License

MIT License (see `LICENSE` file).
