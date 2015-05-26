import numpy
from chainer import cuda, Function

class Parameter(Function):
    """Function that outputs its weight array.

    This is a parameterized function that takes no input and returns a variable
    holding a shallow copy of the parameter array.

    Args:
        array: Initial parameter array.

    """
    parameter_names = 'W',
    gradient_names  = 'gW',

    def __init__(self, array):
        self.W  = array
        self.gW = numpy.empty_like(array)

    def forward(self, x):
        return self.W,

    def backward(self, x, gy):
        self.gW += gy[0]
        return ()
