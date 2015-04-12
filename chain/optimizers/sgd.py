from chain import Optimizer

class SGD(Optimizer):
    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_one(self, param, grad, _):
        param -= self.leraning_rate * grad
