from chainer import cuda
from chainer import optimizer


class SGD(optimizer.GradientMethod):

    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update_one_cpu(self, param, state):
        param.data -= self.lr * param.grad

    def update_one_gpu(self, param, state):
        cuda.elementwise('T grad, T lr', 'T param',
                         'param -= lr * grad',
                         'sgd')(param.grad, self.lr, param.data)
