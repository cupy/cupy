import numpy

from chainer import cuda
from chainer import optimizer


class AdaDelta(optimizer.GradientMethod):

    """Zeiler's ADADELTA.

    See: http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf

    """

    def __init__(self, rho=0.95, eps=1e-6):
        self.rho = rho
        self.eps = eps

    def init_state(self, param, state):
        data = param.data
        xp = cuda.get_array_module(data)
        with cuda.get_device(data):
            state['msg'] = xp.zeros_like(data)
            state['msdx'] = xp.zeros_like(data)

    def update_one_cpu(self, param, state):
        grad = param.grad
        msg, msdx = state['msg'], state['msdx']

        msg *= self.rho
        msg += (1 - self.rho) * grad * grad
        dx = numpy.sqrt((msdx + self.eps) / (msg + self.eps)) * grad
        msdx *= self.rho
        msdx += (1 - self.rho) * dx * dx
        param.data -= dx

    def update_one_gpu(self, param, state):
        cuda.elementwise(
            'T grad, T one_minus_rho, T eps',
            'T param, T msg, T msdx',
            '''msg   = msg + one_minus_rho * (grad * grad - msg);
               T dx  = sqrt((msdx + eps) / (msg + eps)) * grad;
               msdx  += one_minus_rho * (dx * dx - msdx);
               param -= dx;''',
            'adadelta')(param.grad, 1 - self.rho, self.eps,
                        param.data, state['msg'], state['msdx'])
