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
            'T grad, T lr, T momentum',
            'T param, T v',
            '''v = momentum * v - lr * grad;
               param += v;''',
            'momentum_sgd')(grad, self.lr, self.momentum,
                            param, v)
