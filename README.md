# chainer: Neural network framework

## Requirements

Minimum requirements:
- Python 2.7+ (v2.7.9 is recommended)
- NumPy

Requirements to enable CUDA:
- CUDA 6.5+
- PyCUDA
- scikits.cuda (pip install 'scikits.cuda>=0.5.0b1,!=0.042')
- Mako (depending through PyCUDA)

Recommended:
- CuDNN v2
- scikit-learn (to run some examples)
- OpenCV 2.4 (to run some examples)

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
python setup_cuda_deps.py install
```

## License

MIT License (see `LICENSE` file).
