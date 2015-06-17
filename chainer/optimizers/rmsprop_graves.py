import numpy
from chainer import cuda, Optimizer

class RMSpropGraves(Optimizer):
    """Alex Graves's RMSprop.

    See http://arxiv.org/abs/1308.0850

    """

    def __init__(self, lr=1e-4, alpha=0.95, momentum=0.9, eps=1e-4):
        # Default parameter values are the ones in the original paper.
        self.lr    = lr
        self.alpha = alpha
        self.eps   = eps
        self.momentum = momentum

    def init_state_cpu(self, param, grad):
        # n, g, delta
        return numpy.zeros_like(param), numpy.zeros_like(param), numpy.zeros_like(param)

    def init_state_gpu(self, param, grad):
        # n, g, delta
        return cuda.zeros_like(param), cuda.zeros_like(param), cuda.zeros_like(param)

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
            '''float* param, const float* grad, float* avg_n, float* avg_g, float* delta,
               float lr, float alpha, float momentum, float eps''',
            '''avg_n[i] = alpha * avg_n[i] + (1 - alpha) * grad[i] * grad[i];
               avg_g[i] = alpha * avg_g[i] + (1 - alpha) * grad[i];
               delta[i] = delta[i] * momentum - lr * grad[i] * rsqrtf(avg_n[i] - avg_g[i] * avg_g[i] + eps);
               param[i] += delta[i];''',
            'rmsprop_graves')(param, grad, n, g, delta, self.lr, self.alpha, self.momentum, self.eps)
