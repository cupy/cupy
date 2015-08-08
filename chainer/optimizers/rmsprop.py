import numpy

from chainer import cuda
from chainer import optimizer


class RMSprop(optimizer.Optimizer):

    """Hinton's RMSprop."""

    def __init__(self, lr=0.01, alpha=0.99, eps=1e-8):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps

    def init_state_cpu(self, param, grad):
        return numpy.zeros_like(param)

    def init_state_gpu(self, param, grad):
        return cuda.zeros_like(param)

    def update_one_cpu(self, param, grad, ms):
        ms *= self.alpha
        ms += (1 - self.alpha) * grad * grad
        param -= self.lr * grad / (numpy.sqrt(ms) + self.eps)

    def update_one_gpu(self, param, grad, ms):
        cuda.elementwise(
            'T grad, T lr, T alpha, T eps',
            'T param, T ms',
            '''ms = alpha * ms + (1 - alpha) * grad * grad;
               param -= lr * grad / (sqrt(ms) + eps);''',
            'rmsprop')(grad, self.lr, self.alpha, self.eps,
                       param, ms)
