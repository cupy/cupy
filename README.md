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

If you want to enable CUDA, install requirements by:
```
pip install `chainer-cuda-requirements`
```

## Installation from Cloned Repository

You can also install Chainer from the source code:
```
python setup.py install
```

## License

MIT License (see `LICENSE` file).
