import numpy

import cuda
from function import Function

class FunctionSet(object):
    """Manager of a set of functions.

    User typically stores parameterized functions into FunctionSet. FunctionSet
    makes it easy to controll cpu/gpu migration and manage the list of
    parameters/gradients.

    """
    def __init__(self, **functions):
        for name, func in functions.iteritems():
            setattr(self, name, func)

    def collect_parameters(self):
        """Collect parameters and gradients."""
        return self.parameters, self.gradients

    def to_gpu(self, device=None):
        """Move all parameters and gradients to GPU."""
        for func in self.__dict__.itervalues():
            func.to_gpu(device=device)

    def to_cpu(self):
        """Move all parameters and gradients to CPU."""
        for func in self.__dict__.itervalues():
            func.to_cpu()

    @property
    def parameters(self):
        return sum((func.parameters for _, func in self._get_sorted_funcs()), ())

    @parameters.setter
    def parameters(self, params):
        param_iter = iter(params)
        for _, func in self._get_sorted_funcs():
            func.parameters = param_iter

    @property
    def gradients(self):
        return sum((func.gradients for _, func in self._get_sorted_funcs()), ())

    @gradients.setter
    def gradients(self, grads):
        grad_iter = iter(grads)
        for _, func in self._get_sorted_funcs():
            func.gradients = grad_iter

    def _get_sorted_funcs(self):
        return sorted(self.__dict__.iteritems())
