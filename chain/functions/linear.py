import math
import numpy
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel
from pytools import memoize
import scikits.cuda.linalg as culinalg
import scikits.cuda.misc as cumisc
from chain import Function

@memoize
def _add_bias_kernel():
    # TODO(beam2d): More efficient kernel
    return ElementwiseKernel(
        'float* y, float* b, int n_channel',
        'y[i] += b[i % n_channel]')

def add_bias(mat, vec):
    kernel = _add_bias_kernel()
    kernel(mat, vec, vec.size)

class Linear(Function):
    """Implementation of fully-connected layer."""

    parameter_names = ('W', 'b')
    gradient_names  = ('gW', 'gb')

    def __init__(self, in_size, out_size, wscale=1, bias=0):
        self.W = numpy.random.normal(
            0, wscale * math.sqrt(1. / in_size),
            (out_size, in_size)).astype(numpy.float32)
        self.b = numpy.repeat(numpy.float32(bias), out_size)

        self.gW = numpy.empty_like(self.W)
        self.gb = numpy.empty_like(self.b)

    def forward_cpu(self, x):
        return x[0].dot(self.W.T) + self.b,

    def forward_gpu(self, x):
        Wx = culinalg.dot(x[0], self.W, transb='T')
        add_bias(Wx, self.b)
        return Wx,

    def backward_cpu(self, x, gy):
        self.gW += gy[0].T.dot(x[0])
        self.gb += gy[0].sum(0)
        return gy[0].dot(self.W),

    def backward_gpu(self, x, gy):
        culinalg.add_dot(gy[0], x[0], self.gW, transa='T')
        self.gb += cumisc.sum(gy[0], 0)
        return culinalg.dot(gy[0], self.W),
