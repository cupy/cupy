import numpy

from chainer import initializer


class Identity(initializer.Initializer):

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, shape):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('Identity matrix initialization can only be used '
                             'for 2D square matrices.')
        return scale * numpy.identity(shape[0])


class Constant(initializer.Initializer):

    def __init__(self, fill_value):
        self.fill_value = fill_value:

    def __call__(self, shape):
        numpy.full(shape, fill_value)


def zero(shape):
    return Constant(0.0)(shape)


def one(shape):
    return Constant(1.0)(shape)
