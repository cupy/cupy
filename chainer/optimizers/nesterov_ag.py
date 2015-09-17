import numpy

from chainer import cuda
from chainer import optimizer


class NesterovAG(optimizer.Optimizer):

    """Nesterov's Accelarated Gradient.

    Formulated as the linear combination coefficients of the velocity and
    gradient contributions at each iteration.

    See: http://arxiv.org/abs/1212.0901
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def init_state_cpu(self, param, grad):
        return numpy.zeros_like(param)

    def init_state_gpu(self, param, grad):
        return cuda.zeros_like(param)

    def update_one_cpu(self, param, grad, v):
        v *= self.momentum
        v -= self.lr * grad
        param += self.momentum * self.momentum * v
        param -= (1 + self.momentum) * self.lr * grad

    def update_one_gpu(self, param, grad, v):
        cuda.elementwise(
            'T grad, T lr, T momentum',
            'T param, T v',
            '''v = v * momentum - lr * grad;
               param += momentum * momentum * v;
               param -= (1 + momentum) * lr * grad;''',
            'nesterov_ag')(grad, self.lr, self.momentum,
                           param, v)
