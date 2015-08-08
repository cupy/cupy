import numpy

from chainer import cuda
from chainer import optimizer


class AdaDelta(optimizer.Optimizer):

    """Zeiler's ADADELTA.

    See: http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf

    """

    def __init__(self, rho=0.95, eps=1e-6):
        self.rho = rho
        self.eps = eps

    def init_state_cpu(self, param, grad):
        return numpy.zeros_like(param), numpy.zeros_like(param)

    def init_state_gpu(self, param, grad):
        return cuda.zeros_like(param), cuda.zeros_like(param)

    def update_one_cpu(self, param, grad, state):
        msg, msdx = state
        msg *= self.rho
        msg += (1 - self.rho) * grad * grad
        dx = numpy.sqrt((msdx + self.eps) / (msg + self.eps)) * grad
        msdx *= self.rho
        msdx += (1 - self.rho) * dx * dx
        param -= dx

    def update_one_gpu(self, param, grad, state):
        msg, msdx = state
        cuda.elementwise(
            'T grad, T one_minus_rho, T eps',
            'T param, T msg, T msdx',
            '''msg   = msg + one_minus_rho * (grad * grad - msg);
               T dx  = sqrt((msdx + eps) / (msg + eps)) * grad;
               msdx  += one_minus_rho * (dx * dx - msdx);
               param -= dx;''',
            'adadelta')(grad, 1 - self.rho, self.eps,
                        param, msg, msdx)
