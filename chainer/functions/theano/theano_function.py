import collections

import numpy
import six

try:
    import theano
    _available = True
except ImportError:
    _available = False

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _to_var_tuple(vs):
    msg = ('inputs and outputs must be a TensorVariable, a list '
           'of TensorVariable or a tuple of TensorVariable')

    if isinstance(vs, theano.tensor.TensorVariable):
        return vs,
    elif isinstance(vs, collections.Iterable):
        vs = tuple(vs)
        if not all(isinstance(v, theano.tensor.TensorVariable) for v in vs):
            raise TypeError(msg)
        return vs
    else:
        raise TypeError(msg)


class TheanoFunction(function.Function):

    """Theano function wrapper.

    This function wrapps Theano function as a :class:`chainer.Function`.
    A user needs to make input Theano variables and output Theano variables.
    This function automatically creates Theano function for forward calculation
    and backward calculation from inputs and ouptuts. And then, it sends data
    in :class:`chainer.Variable` to the function and gets results from Theano.

    If a user want to use GPUs, he/she can directly send GPU data to Theano
    function without copying.

    .. admonition:: Example

       >>> x = theano.tensor.fvector()
       >>> y = theano.tensor.fvector()
       >>> z = x + y
       >>> w = x - y
       >>> f = F.TheanoFunction(
       ...     inputs=[x, y], outputs=[z, w], optimize_gpu=False)
       >>> a = chainer.Variable(numpy.array([1, 2], dtype='f'))
       >>> b = chainer.Variable(numpy.array([2, 3], dtype='f'))
       >>> c, d = f(a, b)
       >>> c.data
       array([ 3.,  5.], dtype=float32)
       >>> d.data
       array([-1., -1.], dtype=float32)

    Args:
        inputs (tuple of ~theano.tensor.TensorVariable): Input variables of
            Theano. This function accepts the same number of
            :class:`~chainer.Variable`s in forward computation.
        outputs (tuple of ~theano.tensor.TensorVariable): Output variables of
            Theano. The function returns the same number
            :class:`~chainder.Variable`s as ``outputs``.

    """

    def __init__(self, inputs, outputs):
        if not _available:
            msg = '''theano is not installed on your environment.
Please install theano to activate theano function.

  $ pip install theano'''
            raise RuntimeError(msg)

        self._inputs = _to_var_tuple(inputs)
        self._outputs = _to_var_tuple(outputs)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == len(self._inputs))

        for actual_type, variable in six.moves.zip(in_types, self._inputs):
            expect_type = variable.type
            # Theano cannot check shapes of variables
            type_check.expect(
                actual_type.ndim == expect_type.ndim,
                actual_type.dtype == expect_type.numpy_dtype,
            )

    def forward(self, inputs):
        gpu = cuda.get_array_module(*inputs) is not numpy
        # TODO(unno): We can remove redundant gpu-cpu copy using
        # theano.sandbox.cuda.basic_ops.gpu_from_host
        outs = self._outputs
        inputs = [cuda.to_cpu(x) for x in inputs]

        self.func = theano.function(inputs=self._inputs, outputs=outs)
        outputs = self.func(*inputs)

        if gpu:
            # TODO(unno): We can remove redundant gpu-cpu copy using
            # theano.sandbox.cuda.CudaNdarray.gpudata
            device = cuda.get_device(inputs)
            outputs = [cuda.to_gpu(x, device) for x in outputs]

        return tuple(outputs)

    def backward(self, inputs, grads):
        args = inputs + grads
        gpu = cuda.get_array_module(*args) is not numpy

        gs = tuple(
            o.type('g_{}'.format(i)) for i, o in enumerate(self._outputs))
        known_grads = dict(zip(self._outputs, gs))

        # TODO(unno): We can remove redundant gpu-cpu copy using
        # theano.sandbox.cuda.basic_ops.gpu_from_host
        args = [cuda.to_cpu(x) for x in args]

        grad = theano.tensor.grad(
            cost=None, wrt=self._inputs, known_grads=known_grads,
            disconnected_inputs='ignore')

        grad_func = theano.function(
            inputs=self._inputs + gs,
            outputs=grad,
            on_unused_input='ignore')

        outputs = grad_func(*args)
        assert len(outputs) == len(inputs)

        if gpu:
            # TODO(unno): We can remove redundant gpu-cpu copy using
            # theano.sandbox.cuda.CudaNdarray.gpudata
            device = cuda.get_device(inputs)
            outputs = [cuda.to_gpu(x, device) for x in outputs]

        results = []
        for o, i in zip(outputs, inputs):
            if i.dtype.kind != 'f':
                o = None
            elif o.dtype != i.dtype:
                o = o.astype(i.dtype)
            results.append(o)
        return tuple(results)
