class FunctionSet(object):
    """Manager of a set of functions.

    User typically stores parameterized functions into FunctionSet. FunctionSet
    makes it easy to controll cpu/gpu migration and manage the list of
    parameters/gradients.

    """
    def __init__(self, **functions):
        self.functions = {}
        for name, func in functions.iteritems():
            setattr(self, name, func)

    def __setitem__(self, name, func):
        assert type(func) == Function
        self.functions[name] = func

    def __getitem__(self, name):
        return self.functions[name]

    def collect_parameters(self):
        """Collect parameters and gradients."""
        params, grads = [], []
        for func in self.functions.itervalues():
            params += func.parameters
            grads  += func.gradients
        return params, grads

    def to_gpu(self):
        """Move all parameters and gradients to GPU."""
        for func in self.functions.itervalues():
            params = func.parameters
            func.parameters = (self._to_gpu(w) for w in params)
            grads  = func.gradients
            func.gradients  = (self._to_gpu(g) for g in grads)

    def to_cpu(self):
        """Move all parameters and gradients to CPU."""
        for func in self.functions.itervalues():
            params = func.parameters
            func.parameters = (_to_cpu(w) for w in params)
            grads  = func.gradients
            func.gradients  = (_to_cpu(g) for g in grads)
