import numpy

from chainer import cuda
from chainer import optimizer


class AdaGrad(optimizer.Optimizer):

    """AdaGrad implementation.

    See: http://jmlr.org/papers/v12/duchi11a.html

    """

    def __init__(self, lr=0.001, eps=1e-8):
        self.lr = lr
        self.eps = eps

    def init_state_cpu(self, param, grad):
        return numpy.zeros_like(param)

    def init_state_gpu(self, param, grad):
        return cuda.zeros_like(param)

    def update_one_cpu(self, param, grad, h):
        h += grad * grad
        param -= self.lr * grad / (numpy.sqrt(h) + self.eps)

    def update_one_gpu(self, param, grad, h):
        cuda.elementwise(
            'T grad, T lr, T eps',
            'T param, T h',
            '''h += grad * grad;
               param -= lr * grad / (sqrt(h) + eps);''',
            'adagrad')(grad, self.lr, self.eps,
                       param, h)
