import numpy
import pycuda.curandom as curandom
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel
from pytools import memoize
from chainer import Function
from chainer.random import get_generator

@memoize
def _dropout_kernel():
    return ElementwiseKernel(
        'float* y, const float* x, const float* rand, float dropout_ratio, float scale',
        'y[i] = rand[i] < dropout_ratio ? 0 : scale * x[i]')

class Dropout(Function):
    """Dropout regularization."""

    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def forward_cpu(self, x):
        scale = numpy.float32(1. / (1 - self.dropout_ratio))
        self.mask = scale * (numpy.random.rand(*x[0].shape) >= self.dropout_ratio)
        return x[0] * self.mask,

    def forward_gpu(self, x):
        gen = get_generator()
        self.rand = gen.gen_uniform(x[0].shape, dtype=numpy.float32)
        self.scale = 1. / (1 - self.dropout_ratio)

        y = gpuarray.empty_like(x[0])
        _dropout_kernel()(y, x[0], self.rand, self.dropout_ratio, self.scale)
        return y,

    def backward_cpu(self, x, gy):
        return gy[0] * self.mask,

    def backward_gpu(self, x, gy):
        gx = gpuarray.empty_like(gy[0])
        _dropout_kernel()(gx, gy[0], self.rand, self.dropout_ratio, self.scale)
        return gx,


def dropout(x, ratio=.5, train=True):
    if train:
        return Dropout(ratio)(x)
    return x
