import numpy
from chainer import cuda, Function

class Parameter(Function):
    """Function that returns its weight array as an output variable."""

    parameter_names = 'W',
    gradient_names  = 'gW',

    def __init__(self, array):
        self.W  = array
        self.gW = numpy.empty_like(array)

    def forward(self, x):
        return self.W,

    def backward_cpu(self, x, gy):
        self.gW[:] += gy[0]
        return ()

    def backward_gpu(self, x, gy):
        self.gW += gy[0]
        return ()
