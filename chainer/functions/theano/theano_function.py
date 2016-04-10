import collections

import numpy
import theano

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _make_var_tuple(vs):
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

    def __init__(self, inputs, outputs):
        inputs = _make_var_tuple(inputs)
        outputs = _make_var_tuple(outputs)

        self.func = theano.function(inputs=inputs, outputs=outputs)
        gs = tuple(o.type('g_{}'.format(i)) for i, o in enumerate(outputs))
        know_grads = dict(zip(outputs, gs))

        grad = theano.tensor.grad(
            cost=None, wrt=inputs, known_grads=know_grads,
            disconnected_inputs='ignore')
        self.grad = theano.function(
            inputs=inputs + gs,
            outputs=grad,
            on_unused_input='ignore')

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == len(self.func.input_storage))

        for i, storage in enumerate(self.func.input_storage):
            expect_type = storage.type
            actual_type = in_types[i]
            # Theano cannot check shapes of variables
            type_check.expect(
                actual_type.ndim == expect_type.ndim,
                actual_type.dtype == expect_type.numpy_dtype,
            )

    def forward(self, inputs):
        gpu = cuda.get_array_module(*inputs) is not numpy
        if gpu:
            inputs = [cuda.to_cpu(x) for x in inputs]

        outputs = self.func(*inputs)

        if gpu:
            outputs = [cuda.to_gpu(x) for x in outputs]
        return tuple(outputs)

    def backward(self, inputs, grads):
        args = inputs + grads
        gpu = cuda.get_array_module(*args) is not numpy
        if gpu:
            args = [cuda.to_cpu(x) for x in args]

        outs = self.grad(*args)
        assert len(outs) == len(inputs)

        outputs = []
        for o, i in zip(outs, inputs):
            if i.dtype.kind != 'f':
                o = None
            elif o.dtype != i.dtype:
                o = o.astype(i.dtype)
            outputs.append(o)

        if gpu:
            outputs = [cuda.to_gpu(x) if x is not None else None
                       for x in outputs]
        return tuple(outputs)
