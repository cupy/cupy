import collections

import numpy
import six

try:
    import theano
    import theano.sandbox.cuda as theano_cuda
    _available = True
except ImportError:
    _available = False

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

    def __init__(self, inputs, outputs, gpu=True):
        if not _available:
            msg = '''theano is not installed on your environment.
Please install theano to activate theano function.

  $ pip install theano'''
            raise RuntimeError(msg)

        inputs = _make_var_tuple(inputs)
        outputs = _make_var_tuple(outputs)
        if gpu:
            outs = tuple(theano.sandbox.cuda.basic_ops.gpu_from_host(o)
                         if o.type.dtype == 'float32' else o for o in outputs)
        else:
            outs = outputs

        self.func = theano.function(inputs=inputs, outputs=outs)
        gs = tuple(o.type('g_{}'.format(i)) for i, o in enumerate(outputs))
        known_grads = dict(zip(outputs, gs))

        grad = theano.tensor.grad(
            cost=None, wrt=inputs, known_grads=known_grads,
            disconnected_inputs='ignore')

        if gpu:
            grad = tuple(theano.sandbox.cuda.basic_ops.gpu_from_host(g)
                         if g.type.dtype == 'float32' else g for g in grad)

        self.grad = theano.function(
            inputs=inputs + gs,
            outputs=grad,
            on_unused_input='ignore')

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == len(self.func.input_storage))

        for actual_type, storage in six.moves.zip(in_types, self.func.input_storage):
            expect_type = storage.type
            # Theano cannot check shapes of variables
            type_check.expect(
                actual_type.ndim == expect_type.ndim,
                actual_type.dtype == expect_type.numpy_dtype,
            )

    def forward(self, inputs):
        gpu = cuda.get_array_module(*inputs) is not numpy
        if gpu:
            inputs = [_make_theano_array(x) for x in inputs]

        outputs = self.func(*inputs)

        if gpu:
            outputs = [_make_cupy_array(x) for x in outputs]
        return tuple(outputs)

    def backward(self, inputs, grads):
        args = inputs + grads
        gpu = cuda.get_array_module(*args) is not numpy
        if gpu:
            args = [_make_theano_array(x) for x in args]

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
            outputs = [_make_cupy_array(x) for x in outputs]
        return tuple(outputs)


def _make_theano_array(x):
    if isinstance(x, cuda.cupy.ndarray) and x.dtype == numpy.float32:
        return _cupy_to_theano_array(x)
    else:
        return cuda.to_cpu(x)


def _cupy_to_theano_array(x):
    if six.PY2:
        ptr = long(x.data.ptr)  # NOQA
    else:
        ptr = int(x.data.ptr)
    strides = [s // 4 for s in x.strides]
    return theano_cuda.from_gpu_pointer(ptr, x.shape, strides, x)


class CudaNdarrayMemory(object):

    def __init__(self, array):
        self._array = array
        self.device = cuda.cupy.cuda.Device()
        self.ptr = array.gpudata


def _theano_to_cupy_array(x):
    mem = CudaNdarrayMemory(x)
    memptr = cuda.cupy.cuda.MemoryPointer(mem, 0)
    return cuda.cupy.ndarray(x.shape, dtype=numpy.float32, memptr=memptr)


def _make_cupy_array(x):
    if x is None:
        return None
    elif isinstance(x, theano_cuda.CudaNdarray):
        return _theano_to_cupy_array(x)
    else:
        return cuda.to_gpu(x)
