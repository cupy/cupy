import numpy

from chainer import function
from chainer.utils import type_check


class Parameter(function.Function):

    """Function that outputs its weight array.

    This is a parameterized function that takes no input and returns a variable
    holding a shallow copy of the parameter array.

    Args:
        array: Initial parameter array.

    """
    parameter_names = 'W',
    gradient_names = 'gW',

    def __init__(self, array):
        self.W = array
        self.gW = numpy.full_like(array, numpy.nan)

    def __call__(self, volatile=False):
        ret = super(Parameter, self).__call__()
        if volatile:
            ret.unchain_backward()
        ret.volatile = volatile
        return ret

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 0)

    def forward(self, x):
        return self.W,

    def backward(self, x, gy):
        self.gW += gy[0]
        return ()
