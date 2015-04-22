import math
import numpy
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from pytools import memoize
import scikits.cuda.linalg as culinalg
import scikits.cuda.misc as cumisc
from chainer import Function

@memoize
def _add_bias_kernel():
    return ElementwiseKernel('float* y, float* b, int n_channel',
                             'y[i] += b[i % n_channel]')

class Linear(Function):
    """Implementation of fully-connected layer."""

    def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False):
        self.W = numpy.random.normal(
            0, wscale * math.sqrt(1. / in_size),
            (out_size, in_size)).astype(numpy.float32)
        self.gW = numpy.empty_like(self.W)

        if nobias:
            self.b  = None
            self.gb = None
        else:
            self.b  = numpy.repeat(numpy.float32(bias), out_size)
            self.gb = numpy.empty_like(self.b)

    @property
    def parameter_names(self):
        if self.b is None:
            return 'W',
        return 'W', 'b'

    @property
    def gradient_names(self):
        if self.gb is None:
            return 'gW',
        return 'gW', 'gb'

    def forward_cpu(self, x):
        Wx = x[0].dot(self.W.T)
        if self.b is not None:
            Wx += self.b
        return Wx,

    def forward_gpu(self, x):
        Wx = culinalg.dot(x[0], self.W, transb='T')
        if self.b is not None:
            _add_bias_kernel()(Wx, self.b, self.b.size)
        return Wx,

    def backward_cpu(self, x, gy):
        self.gW += gy[0].T.dot(x[0])
        if self.gb is not None:
            self.gb += gy[0].sum(0)
        return gy[0].dot(self.W),

    def backward_gpu(self, x, gy):
        culinalg.add_dot(gy[0], x[0], self.gW, transa='T')
        if self.gb is not None:
            self.gb += cumisc.sum(gy[0], 0)
        return culinalg.dot(gy[0], self.W),
