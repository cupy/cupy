import numpy

from chainer import initializer


class Uniform(initializer.Initializer):

    def __init__(self, scale=0.05):
        self.scale = scale

    def __call__(self, shape):
        return numpy.random.uniform(
            low=-self.scale, high=self.scale, size=shape)


class LeCunUniform(initializer.Initializer):
    '''Reference: LeCun 98, Efficient Backprop

    http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

    '''

    def __init__(self. scale=1.0):
        self.scale = scale

    def __call__(self, shape):
        fan_in, fan_out = initializer.get_fans(shape)
        s = self.scale * numpy.sqrt(3. / fan_in)
        return uniform(shape, s)


class GlorotUniform(initializer.Initializer):

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, shape):
        fan_in, fan_out = initializer.get_fans(shape)
        s = self.scale * np.sqrt(6. / (fan_in + fan_out))
        return uniform(shape, s)


class HeUniform(initializer.Initializer):

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, shape):
        fan_in, fan_out = initializer.get_fans(shape)
        s = self.scale * np.sqrt(6. / fan_in)
        return uniform(shape, s)


