import numpy

from chainer import cuda
from chainer import optimizer


class SMORMS3(optimizer.GradientMethod):

    """Simon Funk's SMORMS3.

    See http://sifter.org/~simon/journal/20150420.html.

    """

    def __init__(self, lr=0.001, eps=1e-16):
        self.lr = lr
        self.eps = eps

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['mem'] = xp.ones_like(param.data)
            state['g'] = xp.zeros_like(param.data)
            state['g2'] = xp.zeros_like(param.data)

    def update_one_cpu(self, param, state):
        mem, g, g2 = state['mem'], state['g'], state['g2']
        grad = param.grad

        r = 1 / (mem + 1)
        g = (1 - r) * g + r * grad
        g2 = (1 - r) * g2 + r * grad * grad
        x = g * g / (g2 + self.eps)
        param.data -= grad * numpy.minimum(x, self.lr) \
            / (numpy.sqrt(g2) + self.eps)
        mem = 1 + mem * (1 - x)

        state['mem'], state['g'], state['g2'] = mem, g, g2

    def update_one_gpu(self, param, state):
        cuda.elementwise(
            'T grad, T lr, T eps',
            'T param, T mem, T g, T g2',
            '''T r, x;
               r = 1 / (mem + 1);
               g = (1 - r) * g + r * grad;
               g2 = (1 - r) * g2 + r * grad * grad;
               x = g * g / (g2 + eps);
               param -= grad * min(lr, x) / (sqrt(g2) + eps);
               mem = 1 + mem * (1 - x)
               ''',
            'smorms3')(param.grad, self.lr, self.eps,
                       param.data, state['mem'], state['g'], state['g2'])
