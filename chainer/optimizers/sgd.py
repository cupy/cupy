from chainer import Optimizer

class SGD(Optimizer):
    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update_one(self, param, grad, _):
        param -= self.lr * grad
