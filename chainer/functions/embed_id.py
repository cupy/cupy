import numpy
from pycuda.elementwise import ElementwiseKernel
from pycuda import gpuarray
from pytools import memoize
from chainer import Function

@memoize
def _lookup_kernel():
    return ElementwiseKernel(
        'float* y, const float* W, const int* x, int n_out',
        'y[i] = W[x[i / n_out] * n_out + i % n_out]')

@memoize
def _grad_kernel():
    # TODO(beam2d): Better kernel
    return ElementwiseKernel(
        'const float* gy, float* gW, const int* x, int n_out',
        'atomicAdd(gW + x[i / n_out] * n_out + i % n_out, gy[i])')

class EmbedID(Function):
    """Efficient linear function for one-hot input."""

    parameter_names = ('W',)
    gradient_names  = ('gW',)

    def __init__(self, in_size, out_size):
        self.W  = numpy.random.randn(in_size, out_size).astype(numpy.float32)
        self.gW = numpy.empty_like(self.W)

    def forward_cpu(self, x):
        return self.W[x[0]],

    def forward_gpu(self, x):
        y = gpuarray.empty((x[0].size, self.W.shape[1]), dtype=numpy.float32)
        _lookup_kernel()(y, self.W, x[0], self.W.shape[1])
        return y,

    def backward_cpu(self, x, gy):
        numpy.add.at(self.gW, x[0], gy[0])
        return None,

    def backward_gpu(self, x, gy):
        _grad_kernel()(gy[0], self.gW, x[0], self.gW.shape[1])
        return None,
