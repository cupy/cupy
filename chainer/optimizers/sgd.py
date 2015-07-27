from chainer import cuda
from chainer import optimizer


class SGD(optimizer.Optimizer):

    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update_one_cpu(self, param, grad, _):
        param -= self.lr * grad

    def update_one_gpu(self, param, grad, _):
        cuda.elementwise(['param', 'grad', 'lr'],
                         'param[i] -= lr * grad[i]',
                         'sgd')(param, grad, self.lr)
