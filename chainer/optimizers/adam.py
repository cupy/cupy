import math

import numpy

from chainer import cuda
from chainer import optimizer


class Adam(optimizer.Optimizer):

    """Adam optimization algorithm.

    See: http://arxiv.org/abs/1412.6980

    """

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999,
                 lam=1 - 1e-8, eps=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.lam = lam
        self.eps = eps

    def init_state_cpu(self, param, grad):
        return numpy.zeros_like(param), numpy.zeros_like(param)

    def init_state_gpu(self, param, grad):
        return cuda.zeros_like(param), cuda.zeros_like(param)

    def update_one_cpu(self, param, grad, state):
        m, v = state
        m += (1 - self.beta1_t) * (grad - m)
        v += (1 - self.beta2) * (grad * grad - v)
        param -= self.lr * m / (numpy.sqrt(v) + self.eps)

    def update_one_gpu(self, param, grad, state):
        m, v = state
        cuda.elementwise(
            ['param', 'grad', 'm', 'v', 'lr',
             'one_minus_beta1_t', 'one_minus_beta2', 'float eps'],
            '''m[i] += one_minus_beta1_t * (grad[i] - m[i]);
               v[i] += one_minus_beta2 * (grad[i] * grad[i] - v[i]);
               param[i] -= lr * m[i] / (sqrtf(v[i]) + eps);''',
            'adam')(param, grad, m, v, self.lr,
                    1 - self.beta1_t, 1 - self.beta2, self.eps)

    @property
    def lr(self):
        fix1 = 1. - self.beta1 ** self.t
        fix2 = 1. - self.beta2 ** self.t
        return self.alpha * math.sqrt(fix2) / fix1

    @property
    def beta1_t(self):
        return self.beta1 * (self.lam ** (self.t - 1))
