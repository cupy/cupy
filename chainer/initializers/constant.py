import numpy

from chainer import initializer
from chainer import cuda


class Identity(initializer.Initializer):

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, array):
        shape = array.shape
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('Identity matrix initialization can only be used '
                             'for 2D square matrices.')
        array[...] = 0
        xp = cuda.get_array_module(array)
        d = xp.diagonal(array)
        if xp == numpy:
            writeable = d.flags.writeable
            d.flags.writeable = True
        d[...] = self.scale
        if xp == numpy:
            d.flags.writeable = writeable


class Constant(initializer.Initializer):

    def __init__(self, fill_value):
        self.fill_value = fill_value

    def __call__(self, array):
        array[...] = self.fill_value


def Zero():
    return Constant(0.0)


def One():
    return Constant(1.0)
