import math
import numpy
from pycuda import gpuarray
import cuda

def _sqnorm(x):
    if isinstance(x, cuda.GPUArray):
        return float(gpuarray.dot(x, x).get())
    x = x.ravel()
    return float(x.dot(x))

class Optimizer(object):
    """Optimizers' base class."""

    def setup(self, params_grads):
        self.tuples = [(p, g, self.init_state(p, g))
                       for p, g in zip(*params_grads)]
        self.t = 0

    def init_state(self, param, grad):
        if isinstance(param, gpuarray.GPUArray):
            return self.init_state_gpu(param, grad)
        return self.init_state_cpu(param, grad)

    def init_state_cpu(self, param, grad):
        """Initialize state on CPU. Child class using state should override it."""
        return None

    def init_state_gpu(self, param, grad):
        """Initialize state on GPU. Child class using state should override it."""
        return None

    def zero_grads(self):
        """Set gradients zero."""
        for _, g, _ in self.tuples:
            if isinstance(g, cuda.GPUArray):
                with cuda.using_device(g):
                    g.fill(0)
            else:
                g.fill(0)

    def compute_grads_norm(self):
        """Compute norm of the gradient."""
        # TODO(beam2d): Make it asynchronous to CPU when gradients exist on GPU
        sqnorm = 0
        for _, g, _ in self.tuples:
            sqnorm += _sqnorm(g)
        return math.sqrt(sqnorm)

    def clip_grads(self, maxnorm):
        """Clip norm of the gradient."""
        norm = self.compute_grads_norm()
        if norm > maxnorm:
            ratio = maxnorm / norm
            for _, g, _ in self.tuples:
                g *= ratio

    def weight_decay(self, decay):
        """Apply weight decay."""
        for p, g, _ in self.tuples:
            if isinstance(p, cuda.GPUArray):
                with cuda.using_device(p):
                    cuda.elementwise('float* g, const float* p, float decay',
                                     'g[i] -= decay * p[i]',
                                     'weight_decay')(g, p, decay)
            else:
                g -= decay * p

    def update(self):
        self.t += 1
        for p, g, s in self.tuples:
            self.update_one(p, g, s)

    def update_one(self, param, grad, state):
        if isinstance(param, cuda.GPUArray):
            with cuda.using_device(param):
                self.update_one_gpu(param, grad, state)
        else:
            self.update_one_cpu(param, grad, state)

    def update_one_cpu(self, param, grad, state):
        raise NotImplementedError()

    def update_one_gpu(self, param, grad, state):
        raise NotImplementedError()
