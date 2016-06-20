import numpy

from chainer import cuda
from chainer import initializer


# Original code forked from MIT licensed keras project
# https://github.com/fchollet/keras/blob/master/keras/initializations.py

class Normal(initializer.Initializer):

    """Initializes array with a normal distribution.

    Each element of the array is initialized by the value drawn
    independently from Gaussian distribution whose mean is 0,
    and standard deviation is ``scale``.

    Args:
        scale(float): Standard deviation of Gaussian distribution.
    """

    def __init__(self, scale=0.05, **kwargs):
        super(Normal, self).__init__(**kwargs)
        self.scale = scale

    def __call__(self, array=None, shape=None, xp=None):
        if array is None:
            assert isinstance(shape, tuple)
            ret = xp.random.normal(
                loc=0.0, scale=self.scale, size=shape).astype(self.dtype)
            return ret
        assert self.dtype is None or array.dtype == self.dtype
        xp = cuda.get_array_module(array)
        array[...] = xp.random.normal(
            loc=0.0, scale=self.scale, size=array.shape)


class GlorotNormal(initializer.Initializer):

    """Initializes array with scaled Gaussian distribution.

    Each element of the array is initialized by the value drawn
    independently from Gaussian distribution whose mean is 0,
    and standard deviation is
    :math:`scale \\times \\sqrt{\\frac{2}{fan_{in} + fan_{out}}}`,
    where :math:`fan_{in}` and :math:`fan_{out}` are the number of
    input and output units, respectively.

    Reference: Glorot & Bengio, AISTATS 2010

    Args:
        scale (float): A constant that determines the scale
            of the standard deviation.

    """

    def __init__(self, scale=1.0, **kwargs):
        super(GlorotNormal, self).__init__(**kwargs)
        self.scale = scale

    def __call__(self, array=None, shape=None, xp=None):
        if array is None:
            assert isinstance(shape, tuple)
            sh = shape
        else:
            sh = array.shape
        fan_in, fan_out = initializer.get_fans(sh)
        s = self.scale * numpy.sqrt(2. / (fan_in + fan_out))
        return Normal(s, dtype=self.dtype)(array, shape, xp)


class HeNormal(initializer.Initializer):

    """Initializes array with scaled Gaussian distribution.

    Each element of the array is initialized by the value drawn
    independently from Gaussian distribution whose mean is 0,
    and standard deviation is
    :math:`scale \\times \\sqrt{\\frac{2}{fan_{in}}}`,
    where :math:`fan_{in}` is the number of input units.

    Reference:  He et al., http://arxiv.org/abs/1502.01852

    Args:
        scale (float): A constant that determines the scale
            of the standard deviation.

    """

    def __init__(self, scale=1.0, **kwargs):
        super(HeNormal, self).__init__(**kwargs)
        self.scale = scale

    def __call__(self, array=None, shape=None, xp=None):
        if array is None:
            assert isinstance(shape, tuple)
            sh = shape
        else:
            sh = array.shape
        fan_in, fan_out = initializer.get_fans(sh)
        s = self.scale * numpy.sqrt(2. / fan_in)
        return Normal(s, dtype=self.dtype)(array, shape, xp)
