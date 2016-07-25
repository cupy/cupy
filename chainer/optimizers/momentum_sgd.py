from chainer import cuda
from chainer import optimizer


class MomentumSGD(optimizer.GradientMethod):

    """Classical momentum SGD."""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['v'] = xp.zeros_like(param.data)

    def update_one_cpu(self, param, state):
        v = state['v']
        v *= self.momentum
        v -= self.lr * param.grad
        param.data += v

    def update_one_gpu(self, param, state):
        cuda.elementwise(
            'T grad, T lr, T momentum',
            'T param, T v',
            '''v = momentum * v - lr * grad;
               param += v;''',
            'momentum_sgd')(param.grad, self.lr, self.momentum,
                            param.data, state['v'])
