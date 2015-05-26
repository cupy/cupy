# chainer: Neural network framework

## Requirements

Minimum requirements:
- Python 2.7+ (developing on v2.7.9)
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

Install Chainer by
```
pip install -e .
```

If you want to enable CUDA, install requirements by
```
pip install -r cuda-requirements.txt
```
