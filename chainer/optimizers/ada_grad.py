import numpy
from chainer import Optimizer, cuda


class AdaGrad(Optimizer):

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
            'float* param, const float* grad, float* h, float lr, float eps',
            '''h[i] += grad[i] * grad[i];
               param[i] -= lr * grad[i] / (sqrtf(h[i]) + eps);''',
            'adagrad')(param, grad, h, self.lr, self.eps)
