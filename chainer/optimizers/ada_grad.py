import numpy
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from pytools import memoize
from chainer import Optimizer

@memoize
def _update_kernel():
    return ElementwiseKernel(
        'float* param, const flat* grad, float* h, float lr, float eps',
        '''
          h[i] += grad[i] * grad[i];
          param[i] -= lr * grad[i] / (sqrtf(h[i]) + eps);
        ''')

class AdaGrad(Optimizer):
    """AdaGrad implementation.

    See: http://jmlr.org/papers/v12/duchi11a.html

    """
    def __init__(self, lr=0.001, eps=1e-8):
        self.lr  = lr
        self.eps = eps

    def init_state_cpu(self, param, grad):
        return numpy.zeros_like(param)

    def init_state_gpu(self, param, grad):
        return gpuarray.zeros_like(param)

    def update_one_cpu(self, param, grad, h):
        h += grad * grad
        param -= self.lr * grad / (numpy.sqrt(h) + self.eps)

    def update_one_gpu(self, param, grad, h):
        _update_kernel()(param, grad, h, self.lr, self.eps)
