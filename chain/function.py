import copy
import weakref
import numpy
from pycuda.gpuarray import GPUArray

from variable import Variable

def _is_gpu_input(inputs):
    return any(type(x) == GPUArray for x in inputs)

class Function(object):
    """Function node.

    Each function implementation inherits this class. The implementation should
    provides {forward,backward}_{cpu,gpu} methods. By default, backward
    propagates None, which indicates no error propagates back from this node.

    Application of __call__ method inserts a copy of the Function instance to
    the computational graph. Note: the copy is shallow, so be careful to
    implement a function that shares some reference values between multiple
    applications.

    If the implementation holds parameters and corresponding gradients, these
    should be kept as attributes and ``parameter_names`` and ``gradient_names``
    attributes should indicates their names. Then, appropriate accessors are
    defined. Otherwise, the implementation itself must provide these accessors.

    """
    parameter_names = ()
    gradient_names = ()

    def __init__(self):
        self.inputs  = None
        self.outputs = None
        self.rank    = None

    def __call__(self, *inputs):
        """Execute function and chain the input/output variables."""

        # First copy itself to avoid duplication within the graph.
        self = copy.copy(self)

        if any(x.volatile for x in inputs):  # not build graph
            assert all(x.volatile for x in inputs)  # do not mix multiple volatility
            outputs = self.forward(tuple(x.data for x in inputs))
            assert type(outputs) == tuple
            outputs = list(Variable(y, volatile=True) for y in outputs)
            if len(outputs) == 1:
                return outputs[0]
            return outputs

        # Build graph
        # Be careful that forward references must be weak
        self.inputs = list(inputs)
        for i, x in enumerate(inputs):
            if hasattr(x, 'splitter'):
                splitter = x.splitter()
            if not hasattr(x, 'splitter') or splitter is None:
                splitter = Split(x)
                x.splitter = weakref.ref(splitter)
            self.inputs[i] = splitter.add_branch()

        self.rank = max(x.rank for x in self.inputs)

        outputs = self.forward(tuple(x.data for x in self.inputs))
        assert type(outputs) == tuple

        ret = tuple(Variable(y) for y in outputs)
        for y in ret:
            y.set_creator(self)

        # Make forward references weak
        self.outputs = tuple(weakref.ref(y) for y in ret)

        if len(ret) == 1:
            return ret[0]
        return ret

    def forward(self, inputs):
        """Forward function.

        It delegates the procedure to forward_{cpu,gpu} by default. User must
        either implement cpu/gpu methods or override this method.

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
            y_ref = y()
            if y_ref is not None:
                y_ref.creator = None
        for x in self.inputs:
            x.splitter = None
        self.inputs = None

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
        self.outputs.append(weakref.ref(output))
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
