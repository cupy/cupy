from pycuda.elementwise import ElementwiseKernel
from pytools import memoize
from chainer import Optimizer

@memoize
def _update_kernel():
    return ElementwiseKernel('float* param, const float* grad, float lr',
                             'param[i] -= lr * grad[i]')

class SGD(Optimizer):
    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update_one_cpu(self, param, grad, _):
        param -= self.lr * grad

    def update_one_gpu(self, param, grad, _):
        _update_kernel()(param, grad, self.lr)
