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
            ['param', 'grad', 'avg_n', 'avg_g',
             'delta', 'lr', 'alpha', 'momentum', 'eps'],
            '''avg_n[i] = alpha * avg_n[i] + (1 - alpha) * grad[i] * grad[i];
               avg_g[i] = alpha * avg_g[i] + (1 - alpha) * grad[i];
               delta[i] = delta[i] * momentum -
                   lr * grad[i] * rsqrt(avg_n[i] - avg_g[i] * avg_g[i] + eps);
               param[i] += delta[i];''',
            'rmsprop_graves')(param, grad, n, g, delta,
                              param.dtype.type(self.lr),
                              param.dtype.type(self.alpha),
                              param.dtype.type(self.momentum),
                              param.dtype.type(self.eps))
