import numpy

from chainer import cuda
from chainer import optimizer


class AdaGrad(optimizer.GradientMethod):

    """AdaGrad implementation.

    See: http://jmlr.org/papers/v12/duchi11a.html

    """

    def __init__(self, lr=0.001, eps=1e-8):
        self.lr = lr
        self.eps = eps

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['h'] = xp.zeros_like(param.data)

    def update_one_cpu(self, param, state):
        h = state['h']
        grad = param.grad

        h += grad * grad
        param.data -= self.lr * grad / (numpy.sqrt(h) + self.eps)

    def update_one_gpu(self, param, state):
        cuda.elementwise(
            'T grad, T lr, T eps',
            'T param, T h',
            '''h += grad * grad;
               param -= lr * grad / (sqrt(h) + eps);''',
            'adagrad')(param.grad, self.lr, self.eps,
                       param.data, state['h'])
