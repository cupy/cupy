import numpy

from chainer import cuda

if cuda.available:
    import cupy


def as_vec(x):
    return x.reshape(x.size)


def as_mat(x):
    return x.reshape(x.shape[0], x.size // x.shape[0])


def empty_like(x):
    if cuda.available and isinstance(x, cuda.ndarray):
        return cupy.empty_like(x)
    else:
        return numpy.empty_like(x)
