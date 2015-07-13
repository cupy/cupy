# chainer: Neural network framework

## Requirements

Minimum requirements:
- Python 2.7+, 3.4+
- NumPy
- Six 1.9+

Requirements for some features:
- CUDA support
  - CUDA 6.5+
  - PyCUDA
  - scikits.cuda (pip install scikits.cuda>=0.5.0b2,!=0.042)
  - Mako (depending through PyCUDA)
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

If you want to enable CUDA, first you have to install CUDA and set the environment variable `PATH` to enable `nvcc` command.
For example, if you are using Ubuntu and CUDA is installed by the official distribution, then it exists at `/usr/local/cuda/bin`, so you have to add this path to the `PATH` environment.
This is done by adding the following line to `.bashrc` or `.zshrc` (choose which you are using):
```
export PATH=/usr/local/cuda/bin:$PATH
```
Do not forget to restart your terminal session (or `source` it) to enable this change.
Then, install CUDA-related dependent packages via pip:
```
pip install chainer-cuda-deps
```
or, from the source:
```
python cuda_deps/setup.py install
```

## License

MIT License (see `LICENSE` file).
