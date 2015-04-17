import math
import numpy
from pycuda import gpuarray

def _sqnorm(x):
    if type(x) == gpuarray.GPUArray:
        return float(gpuarray.dot(x, x).get())
    x = x.ravel()
    return float(x.dot(x))

class Optimizer(object):
    """Optimizers' base class."""

    def setup(self, params_grads):
        self.tuples = [(p, g, self.init_state(p, g))
                       for p, g in zip(*params_grads)]

    def init_state(self, param, grad):
        """Initialize state. Child class using state should override it. """
        return None

    def zero_grads(self):
        """Set gradients zero."""

        for _, g, _ in self.tuples:
            g.fill(0)

    def clip_grads(self, maxnorm):
        """Clip norm of the gradient."""

        sqnorm = 0
        for _, g, _ in self.tuples:
            sqnorm += _sqnorm(g)
        norm = math.sqrt(sqnorm)
        if norm > maxnorm:
            ratio = maxnorm / norm
            for _, g, _ in self.tuples:
                g *= ratio

    def update(self):
        for p, g, s in self.tuples:
            self.update_one(p, g, s)

    def update_one(self, param, grad, state):
        if type(param) == gpuarray.GPUArray:
            self.update_one_gpu(self, param, grad, state)
        else:
            self.update_one_cpu(self, param, grad, state)

    def update_one_cpu(self, param, grad, state):
        raise NotImplementedError()

    def update_one_gpu(self, param, grad, state):
        raise NotImplementedError()
