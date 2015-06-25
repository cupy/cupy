import numpy

from chainer import cuda
from chainer import optimizer


class MomentumSGD(optimizer.Optimizer):

    """Classical momentum SGD."""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def init_state_cpu(self, param, grad):
        return numpy.zeros_like(param)

    def init_state_gpu(self, param, grad):
        return cuda.zeros_like(param)

    def update_one_cpu(self, param, grad, v):
        v *= self.momentum
        v -= self.lr * grad
        param += v

    def update_one_gpu(self, param, grad, v):
        cuda.elementwise(
            '''float* param, const float* grad, float* v,
               float lr, float momentum''',
            '''v[i] = momentum * v[i] - lr * grad[i];
               param[i] += v[i];''',
            'momentum_sgd')(param, grad, v, self.lr, self.momentum)
