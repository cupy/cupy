import numpy

from chainer import cuda


def as_vec(x):
    return x.ravel()


def as_mat(x):
    return x.reshape(len(x), -1)


def empty_like(x):
    if cuda.available and isinstance(x, cuda.ndarray):
        return cuda.cupy.empty_like(x)
    else:
        return numpy.empty_like(x)
