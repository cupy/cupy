import numpy

from chainer import cuda
from chainer import initializer


# Original code forked from MIT licensed keras project
# https://github.com/fchollet/keras/blob/master/keras/initializations.py

class Uniform(initializer.Initializer):

    """Initializes array with a scaled uniform distribution.

    Each element of the array is initialized by the value drawn
    independently from uniform distribution :math:`[-scale, scale]`.

    Attributes:
        scale (float): A constant that determines the
            scale of uniform distribution.

    """

    def __init__(self, scale=0.05):
        self.scale = scale

    def __call__(self, array):
        xp = cuda.get_array_module(array)
        array[...] = xp.random.uniform(
            low=-self.scale, high=self.scale, size=array.shape)


class LeCunUniform(initializer.Initializer):

    """Initializes array with a scaled uniform distribution.

    Each element of the array is initialized by the value drawn
    independently from uniform distribution :math:`[-s, s]`
    where :math:`s = scale \\times \\sqrt{\\frac{3}{fan\_in}}`.
    Here :math:`fan\_in` is the number of input units.

    Reference: LeCun 98, Efficient Backprop
    http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

    Attributes:
        scale (float): A constant that determines the
            scale of uniform distribution.

    """

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, array):
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(3. / fan_in)
        return Uniform(s)(array)


class GlorotUniform(initializer.Initializer):

    """Initializes array with a scaled uniform distribution.

    Each element of the array is initialized by the value drawn
    independently from uniform distribution :math:`[-s, s]`
    where :math:`s = scale \\times \\sqrt{\\frac{6}{fan\_in + fan\_out}}`.
    Here, :math:`fan\_in` and `fan\_out` are the number of
    input and output units, respectively.

    Attributes:
        scale (float): A constant that determines the
            scale of uniform distribution.

    """

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, array):
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(6. / (fan_in + fan_out))
        return Uniform(s)(array)


class HeUniform(initializer.Initializer):

    """Initializes array with scaled uniform distribution.

    Each element of the array is initialized by the value drawn
    independently from uniform distribution :math:`[-s, s]`
    where :math:`s = scale \\times \\sqrt{\\frac{6}{fan\_in}}`.
    Here, :math:`fan\_in` is the number of input units.

    Attributes:
        scale (float): A constant that determines the
            scale of uniform distribution.

    """

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, array):
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(6. / fan_in)
        return Uniform(s)(array)
