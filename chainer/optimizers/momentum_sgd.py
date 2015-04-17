import numpy
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from pytools import memoize

from chainer import Optimizer

@memoize
def _update_kernel():
    return ElementwiseKernel(
        'float* param, const float* grad, float* v, float lr, float momentum',
        '''
          v[i] = momentum * v[i] - lr * grad[i];
          param[i] += v[i];
        ''')

class MomentumSGD(Optimizer):
    """Classical momentum SGD."""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def init_state_cpu(self, param, grad):
        return numpy.zeros_like(param)

    def init_state_gpu(self, param, grad):
        return gpuarray.zeros_like(param)

    def update_one_cpu(self, param, grad, v):
        v *= self.momentum
        v -= self.lr * grad
        param += v

    def update_one_gpu(self, param, grad, v):
        _update_kernel()(param, grad, v, self.lr, self.momentum)
