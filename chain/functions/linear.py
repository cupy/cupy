import math
import numpy
import pycuda.gpuarray as gpuarray
import scikits.cuda.linalg as culinalg
import scikits.cuda.misc as cumisc

from chain import Function

class Linear(Function):
    """Implementation of fully-connected layer."""

    parameter_names = ('W', 'b')
    gradient_names  = ('gW', 'gb')

    def __init__(self, in_size, out_size, wscale=1):
        self.W = numpy.random.normal(
            0, wscale * math.sqrt(1. / in_size),
            (out_size, in_size)).astype(numpy.float32)
        self.b = numpy.zeros((out_size,), dtype=numpy.float32)

        self.gW = numpy.empty_like(self.W)
        self.gb = numpy.empty_like(self.b)

    def forward_cpu(self, x):
        return x[0].dot(self.W.T) + self.b,

    def forward_gpu(self, x):
        Wx = culinalg.dot(x[0], self.W, transb='T')
        return cumisc.add_matvec(Wx, self.b, axis=1),

    def backward_cpu(self, x, gy):
        self.gW += gy[0].T.dot(x[0])
        self.gb += gy[0].sum(0)
        return gy[0].dot(self.W),

    def backward_gpu(self, x, gy):
        culinalg.add_dot(gy[0], x[0], self.gW, transa='T')
        self.gb += cumisc.sum(gy[0], 0)
        return culinalg.dot(gy[0], self.W),
