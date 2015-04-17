import numpy
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from pytools import memoize
from chainer import Optimizer

@memoize
def _update_kernel():
    return ElementwiseKernel(
        '''
          float* param, const float* grad, float* msg, float* msdx,
          float rho, float eps
        ''', '''
          msg[i]   = rho * msg[i]  + (1 - rho) * grad[i] * grad[i];
          float dx = - sqrtf((msdx + eps) / (msg + eps)) * grad[i];
          msdx[i]  = rho * msdx[i] + (1 - rho) * dx * dx;
          param[i] += dx;
        ''')

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
        return gpuarray.zeros_like(param), gpuarray.zeros_like(param)

    def update_one_cpu(self, param, grad, state):
        msg, msdx = state
        msg *= self.rho
        msg += (1 - self.rho) * grad * grad
        dx = - numpy.sqrt((msdx + self.eps) / (msg + self.eps)) * grad
        msdx *= self.rho
        msdx += (1 - self.rho) * dx * dx
        param += dx

    def update_one_gpu(self, param, grad, ms):
        _update_kernel()(param, grad, msg, msdx, self.rho, self.eps)
