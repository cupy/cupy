import numpy
from pycuda.gpuarray import GPUArray

from variable import Variable

def _is_gpu_input(inputs):
    return any(type(x) == GPUArray for x in inputs)

class Function(object):
    """Function node."""

    def __init__(self):
        self.inputs  = None
        self.outputs = None
        self.rank    = None

    def __call__(self, *inputs):
        """Execute function and chain the input/output variables."""

        self.inputs = list(inputs)
        for i, x in enumerate(inputs):
            if not hasattr(x, 'splitter'):
                x.splitter = Split(x)
            self.inputs[i] = x.splitter.add_branch()

        self.rank = max(x.rank for x in self.inputs)

        outputs = self.forward(tuple(x.data for x in self.inputs))

        self.outputs = list(Variable(y) for y in outputs)
        for y in self.outputs:
            y.set_creator(self)

        if len(self.outputs) == 1:
            return self.outputs[0]
        return self.outputs

    def forward(self, inputs):
        """Forward function.

        It delegates the procedure to forward_{cpu,gpu} by default. User must
        either implement cpu/gpu methods or override this method. It must
        outputs a tuple of variable(s).

        """
        if _is_gpu_input(inputs):
            return self.forward_gpu(inputs)
        return self.forward_cpu(inputs)

    def forward_cpu(self, inputs):
        """Forward function on CPU implemented by child class."""
        raise NotImplementedError()

    def forward_gpu(self, inputs):
        """Forward function on GPU implemented by child class."""
        raise NotImplementedError()

    def backward(self, inputs, grad_outputs):
        if _is_gpu_input(inputs):
            return self.backward_gpu(inputs, grad_outputs)
        return self.backward_cpu(inputs, grad_outputs)

    def backward_cpu(self, inputs, grad_outputs):
        """Default implementation of backward on CPU, which does nothing."""
        return tuple(None for _ in inputs)

    def backward_gpu(self, inputs, grad_outputs):
        """Default implementation of backward on GPU, which does nothing."""
        return tuple(None for _ in inputs)

    def forget(self):
        """Purge in/out variables and remove this node from the graph."""

        for y in self.outputs:
            y.creator = None
        for x in self.inputs:
            x.splitter = None
        self.outputs = None
        self.inputs = None


class Split(Function):
    """Special function to branch the graph at variable node.

    Split does not implement forward: it is intended to implicitly used by
    Function.

    """
    def __init__(self, var):
        self.inputs  = [var]
        self.outputs = []
        self.rank    = var.rank

    def add_branch(self):
        x = self.inputs[0]
        output = Variable(x.data)
        output.set_creator(self)
        self.outputs.append(output)
        return output

    def backward(self, inputs, grad_outputs):
        # Accumulate gradients
        gx = None
        for gy in grad_outputs:
            if gy is None: continue
            if gx is None:
                if len(grad_outputs) == 1:
                    gx = gy  # no copy
                else:
                    # TODO(beam2d): Add fast (no copy) option
                    gx = gy.copy()
            else:
                gx += gy
        return (gx,)


class Layer(object):
    """Parameterized function generator.

    Layer generates parameterized function and attach it to the computational
    graph in __call__. Typical implementation sets parameters and gradients to
    attributes, whose names should be set to parameter_names and gradient_names
    fields. The default implementations of ``parameters`` and ``gradients`` rely
    on them.

    """
    def __call__(self, *a, **kw):
        raise NotImplementedError()

    @property
    def parameters(self):
        return tuple(getattr(self, name) for name in self.parameter_names)

    @parameters.setter
    def parameters(self, values):
        for name, value in zip(self.parameter_names, values):
            setattr(self, name, value)

    @property
    def gradients(self):
        return tuple(getattr(self, name) for name in self.gradient_names)

    @gradients.setter
    def gradients(self, values):
        for name, value in zip(self.gradient_names, values):
            setattr(self, name, value)


def _to_gpu(array):
    if type(array) == gpuarray.GPUArray:
        return array
    return gpuarray.to_gpu(array)

def _to_cpu(array):
    if type(array) == numpy.ndarray:
        return array
    return array.get()

class LayerSet(object):
    """Set of layers (parameterized function generators).

    User typically stores layer instances into LayerSet. LayerSet makes it easy
    to controll cpu vs. gpu and manage the list of parameters/gradients.

    """

    def __init__(self, **layers):
        self.layers = layers
        for name, layer in layers.iteritems():
            setattr(self, name, layer)

    def collect_parameters(self):
        """Collect parameters and gradients."""

        params, grads = [], []
        for layer in self.layers.itervalues():
            params += layer.parameters
            grads  += layer.gradients
        return params, grads

    def to_gpu(self):
        """Move all parameters and gradients to GPU."""

        for layer in self.layers.itervalues():
            params = layer.parameters
            layer.parameters = (_to_gpu(w) for w in params)
            grads  = layer.gradients
            layer.gradients  = (_to_gpu(g) for g in grads)

    def to_cpu(self):
        """Move all parameters and gradients to CPU."""

        for layer in self.layers.itervalues():
            params = layer.parameters
            layer.parameters = (_to_cpu(w) for w in params)
            grads  = layer.gradients
            layer.gradients  = (_to_cpu(g) for g in grads)
