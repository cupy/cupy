import numpy

from chainer import cuda
from chainer import initializer


# Original code forked from MIT licensed keras project
# https://github.com/fchollet/keras/blob/master/keras/initializations.py

class Uniform(initializer.Initializer):

    def __init__(self, scale=0.05):
        self.scale = scale

    def __call__(self, array):
        xp = cuda.get_array_module(array)
        array[...] = xp.random.uniform(
            low=-self.scale, high=self.scale, size=array.shape)


def uniform(array, scale):
    return Uniform(scale)(array)


class LeCunUniform(initializer.Initializer):
    '''Reference: LeCun 98, Efficient Backprop

    http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

    '''

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, array):
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(3. / fan_in)
        return uniform(array, s)


class GlorotUniform(initializer.Initializer):

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, array):
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(6. / (fan_in + fan_out))
        return uniform(array, s)


class HeUniform(initializer.Initializer):

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, array):
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(6. / fan_in)
        return uniform(array, s)
