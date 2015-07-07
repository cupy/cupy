# chainer: Neural network framework

## Requirements

Minimum requirements:
- Python 2.7+, 3.4+
- NumPy
- Six 1.9+

- Requirements for some features:
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

If you want to enable CUDA, first you have to install CUDA and set the environment variable `CUDA_DIR` to the installed path.
You also have to set binary and library paths by appropriate environment variables like `PATH` and `LD_LIBRARY_PATH`.
Then install CUDA-related packages by:
```
pip install chainer-cuda-deps
```

or, from the source:
```
python cuda_deps/setup.py install
```

## License

MIT License (see `LICENSE` file).
