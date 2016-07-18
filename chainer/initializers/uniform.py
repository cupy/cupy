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
            scale of the uniform distribution.
        dtype: Data type specifier.

    """

    def __init__(self, scale=0.05, dtype=None):
        self.scale = scale
        super(Uniform, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        xp = cuda.get_array_module(array)
        array[...] = xp.random.uniform(
            low=-self.scale, high=self.scale, size=array.shape)


class LeCunUniform(initializer.Initializer):

    """Initializes array with a scaled uniform distribution.

    Each element of the array is initialized by the value drawn
    independently from uniform distribution :math:`[-s, s]`
    where :math:`s = scale \\times \\sqrt{\\frac{3}{fan_{in}}}`.
    Here :math:`fan_{in}` is the number of input units.

    Reference: LeCun 98, Efficient Backprop
    http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

    Attributes:
        scale (float): A constant that determines the
            scale of the uniform distribution.
        dtype: Data type specifier.

    """

    def __init__(self, scale=1.0, dtype=None):
        self.scale = scale
        super(LeCunUniform, self).__init__(dtype)

    def __call__(self, array):
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(3. / fan_in)
        Uniform(s)(array)


class GlorotUniform(initializer.Initializer):

    """Initializes array with a scaled uniform distribution.

    Each element of the array is initialized by the value drawn
    independently from uniform distribution :math:`[-s, s]`
    where :math:`s = scale \\times \\sqrt{\\frac{6}{fan_{in} + fan_{out}}}`.
    Here, :math:`fan_{in}` and `fan_{out}` are the number of
    input and output units, respectively.

    Attributes:
        scale (float): A constant that determines the
            scale of the uniform distribution.
        dtype: Data type specifier.

    """

    def __init__(self, scale=1.0, dtype=None):
        self.scale = scale
        super(GlorotUniform, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(6. / (fan_in + fan_out))
        Uniform(s)(array)


class HeUniform(initializer.Initializer):

    """Initializes array with scaled uniform distribution.

    Each element of the array is initialized by the value drawn
    independently from uniform distribution :math:`[-s, s]`
    where :math:`s = scale \\times \\sqrt{\\frac{6}{fan_{in}}}`.
    Here, :math:`fan_{in}` is the number of input units.

    Attributes:
        scale (float): A constant that determines the
            scale of the uniform distribution.
        dtype: Data type specifier.

    """

    def __init__(self, scale=1.0, dtype=None):
        self.scale = scale
        super(HeUniform, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(6. / fan_in)
        Uniform(s)(array)
