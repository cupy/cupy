import numpy


# Original code forked from MIT licensed keras project
# https://github.com/fchollet/keras/blob/master/keras/initializations.py

class Initializer(object):

    def __call__(self, array):
        NotImplementedError()


def get_fans(shape):
    if not isinstance(shape, tuple):
        raise ValueError('shape must be tuple')

    if len(shape) < 2:
        raise ValueError('shape must be of length >= 2: shape={}', shape)

    fan_in = numpy.prod(shape[1:])
    fan_out = shape[0]
    return fan_in, fan_out
