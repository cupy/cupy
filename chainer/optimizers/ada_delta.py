import numpy
from chainer import cuda, Optimizer

class AdaDelta(Optimizer):
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

    def update_one_gpu(self, param, grad, ms):
        cuda.elementwise(
            '''float* param, const float* grad, float* msg, float* msdx,
               float one_minus_rho, float eps''',
            '''msg[i]   += one_minus_rho * (grad[i] * grad[i] - msg[i]);
               float dx  = sqrtf((msdx[i] + eps) / (msg[i] + eps)) * grad[i];
               msdx[i]  += one_minus_rho * (dx * dx - msdx[i]);
               param[i] -= dx;''',
            'adadelta')(param, grad, msg, msdx, 1 - self.rho, self.eps)
