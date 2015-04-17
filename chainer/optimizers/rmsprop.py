import numpy
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from pytools import memoize
from chainer import Optimizer

@memoize
def _update_kernel():
    return ElementwiseKernel(
        '''
          float* param, const float* grad, float* ms,
          float lr, float alpha, float eps
        ''', '''
          ms[i] = alpha * ms[i] + (1 - alpha) * grad[i] * grad[i];
          param[i] -= lr * grad[i] / (sqrtf(ms[i]) + eps);
        ''')

class RMSprop(Optimizer):
    """Hinton's RMSprop."""

    def __init__(self, lr=0.01, alpha=0.99, eps=1e-8):
        self.lr    = lr
        self.alpha = alpha
        self.eps   = eps

    def init_state_cpu(self, param, grad):
        return numpy.zeros_like(param)

    def init_state_gpu(self, param, grad):
        return gpuarray.zeros_like(param)

    def update_one_cpu(self, param, grad, ms):
        ms *= self.alpha
        ms += (1 - self.alpha) * grad * grad
        param -= self.lr * grad / (numpy.sqrt(ms) + self.eps)

    def update_one_gpu(self, param, grad, ms):
        _update_kernel()(param, grad, ms, self.lr, self.alpha, self.eps)
