import numpy

from chainer import cuda
from chainer import optimizer


class RMSpropGraves(optimizer.Optimizer):
    """Alex Graves's RMSprop.

    See http://arxiv.org/abs/1308.0850

    """

    def __init__(self, lr=1e-4, alpha=0.95, momentum=0.9, eps=1e-4):
        # Default parameter values are the ones in the original paper.
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum

    def init_state_cpu(self, param, grad):
        n = numpy.zeros_like(param)
        g = numpy.zeros_like(param)
        delta = numpy.zeros_like(param)
        return n, g, delta

    def init_state_gpu(self, param, grad):
        n = cuda.zeros_like(param)
        g = cuda.zeros_like(param)
        delta = cuda.zeros_like(param)
        return n, g, delta

    def update_one_cpu(self, param, grad, state):
        n, g, delta = state
        n *= self.alpha
        n += (1 - self.alpha) * grad * grad
        g *= self.alpha
        g += (1 - self.alpha) * grad
        delta *= self.momentum
        delta -= self.lr * grad / numpy.sqrt(n - g * g + self.eps)
        param += delta

    def update_one_gpu(self, param, grad, state):
        n, g, delta = state
        cuda.elementwise(
            'T grad, T lr, T alpha, T momentum, T eps',
            'T param, T avg_n, T avg_g, T delta',
            '''avg_n = alpha * avg_n + (1 - alpha) * grad * grad;
               avg_g = alpha * avg_g + (1 - alpha) * grad;
               delta = delta * momentum -
                   lr * grad * rsqrt(avg_n - avg_g * avg_g + eps);
               param += delta;''',
            'rmsprop_graves')(grad,
                              self.lr, self.alpha, self.momentum, self.eps,
                              param, n, g, delta)
