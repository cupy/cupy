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
        object.__setattr__(self, 'functions', {})
        for name, func in functions.iteritems():
            setattr(self, name, func)

    def __getattr__(self, name):
        return self.functions[name]

    def __setattr__(self, name, value):
        if name in ('parameters', 'gradients'):
            object.__setattr__(self, name, value)
        else:
            assert isinstance(value, Function)
            self.functions[name] = value

    def __delattr__(self, name):
        del self.functions[name]

    def collect_parameters(self):
        """Collect parameters and gradients."""
        return self.parameters, self.gradients

    def to_gpu(self):
        """Move all parameters and gradients to GPU."""
        for func in self.functions.itervalues():
            params = func.parameters
            func.parameters = (cuda.to_gpu(w) for w in params)
            grads  = func.gradients
            func.gradients  = (cuda.to_gpu(g) for g in grads)

    def to_cpu(self):
        """Move all parameters and gradients to CPU."""
        for func in self.functions.itervalues():
            params = func.parameters
            func.parameters = (cuda.to_cpu(w) for w in params)
            grads  = func.gradients
            func.gradients  = (cuda.to_cpu(g) for g in grads)

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
        return sorted(self.functions.iteritems())
