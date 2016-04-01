import numpy

from chainer import cuda
from chainer import initializer


class Normal(initializer.Initializer):

    def __init__(self, scale=0.05):
        self.scale = scale

    def __call__(self, array):
        xp = cuda.get_array_module(array)
        array[...] = xp.random.normal(
            loc=0.0, scale=self.scale, size=array.shape)


def normal(array, scale):
    Normal(scale)(array)


class GlorotNormal(initializer.Initializer):
    '''Reference: Glorot & Bengio, AISTATS 2010

    '''

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, array):
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(2. / (fan_in + fan_out))
        return normal(array, s)


class HeNormal(initializer.Initializer):
    '''Reference:  He et al., http://arxiv.org/abs/1502.01852

    '''

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, array):
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(2. / fan_in)
        return normal(array, s)
