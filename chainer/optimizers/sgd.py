from chainer import cuda, Optimizer


class SGD(Optimizer):

    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update_one_cpu(self, param, grad, _):
        param -= self.lr * grad

    def update_one_gpu(self, param, grad, _):
        cuda.elementwise('float* param, const float* grad, float lr',
                         'param[i] -= lr * grad[i]',
                         'sgd')(param, grad, self.lr)
