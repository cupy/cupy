import numpy
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
from pytools import memoize

from chainer import Function, cudnn
from chainer.functions.softmax import Softmax

@memoize
def _cross_entropy_kernel():
    return ReductionKernel(
        numpy.float32,
        arguments='int* t, float* y, int n_channel',
        map_expr='-log(y[i * n_channel + t[i]])',
        neutral='0', reduce_expr='a+b')

@memoize
def _backward_kernel():
    return ElementwiseKernel(
        'float* gx, float* y, int* t, float* gloss, int n_channel',
        'gx[i] = *gloss * (y[i] - ((i % n_channel) == t[i / n_channel]))')

class SoftmaxCrossEntropy(Function):
    """Softmax activation followed by a cross entropy loss."""

    def forward_cpu(self, inputs):
        x, t = inputs
        self.y, = Softmax().forward_cpu((x,))
        return -numpy.log(self.y[xrange(len(t)), t]).sum(keepdims=True),

    def forward_gpu(self, inputs):
        x, t = inputs
        self.y, = Softmax().forward_gpu((x,))
        return _cross_entropy_kernel()(t, self.y, self.y.shape[1]),

    def backward_cpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        gx = self.y  # reuse the memory
        gx[xrange(len(t)), t] -= 1
        gx *= gloss[0]
        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        gx = self.y  # reuse the memory
        _backward_kernel()(gx, self.y, t, gloss, self.y.shape[1])
        return gx, None


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)
