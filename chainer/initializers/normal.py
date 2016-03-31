import numpy

from chainer import initializer


class Normal(initializer.Initializer):

    def __init__(self, scale=0.05):
        self.scale = scale

    def __call__(self, shape):
        return numpy.random.normal(
            loc=0.0, scale=self.scale, size=shape)


def normal(shape, scale):
    return Normal(scale)(shape)


class GlorotNormal(initializer.Initializer):
    '''Reference: Glorot & Bengio, AISTATS 2010

    '''

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, shape):
        fan_in, fan_out = initializer.get_fans(shape)
        s = self.scale * numpy.sqrt(2. / (fan_in + fan_out))
        return normal(shape, s)


class HeNormal(initializer.Initializer):
    '''Reference:  He et al., http://arxiv.org/abs/1502.01852

    '''

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, shape):
        fan_in, fan_out = initializer.get_fans(shape)
        s = self.scale * numpy.sqrt(2. / fan_in)
        return normal(shape, s)


