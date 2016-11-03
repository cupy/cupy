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
        optimize_gpu (bool): If True, remove CPU/GPU copying as much as
            possible. If False, it always copies :class:`cupy.ndarray` to CPU
            before passing it to Theano. Note that this feature only supports
            ``float32`` data.

    """

    def __init__(self, inputs, outputs, optimize_gpu=True):
        if not _available:
            msg = '''theano is not installed on your environment.
Please install theano to activate theano function.

  $ pip install theano'''
            raise RuntimeError(msg)

        self._inputs = _to_var_tuple(inputs)
        self._outputs = _to_var_tuple(outputs)
        self.optimize_gpu = optimize_gpu

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
        if gpu and self.optimize_gpu:
            outs = tuple(theano.sandbox.cuda.basic_ops.gpu_from_host(o)
                         if o.dtype == 'float32' else o for o in self._outputs)
            inputs = [_cupy_array_to_theano_input(x) for x in inputs]
        else:
            outs = self._outputs

        self.func = theano.function(inputs=self._inputs, outputs=outs)
        outputs = self.func(*inputs)

        if gpu:
            device = theano.sandbox.cuda.active_device_number()
            outputs = [_theano_output_to_cupy_array(x, device)
                       for x in outputs]
        return tuple(outputs)

    def backward(self, inputs, grads):
        args = inputs + grads
        gpu = cuda.get_array_module(*args) is not numpy

        gs = tuple(
            o.type('g_{}'.format(i)) for i, o in enumerate(self._outputs))
        known_grads = dict(zip(self._outputs, gs))

        grad = theano.tensor.grad(
            cost=None, wrt=self._inputs, known_grads=known_grads,
            disconnected_inputs='ignore')

        if gpu and self.optimize_gpu:
            grad = tuple(theano.sandbox.cuda.basic_ops.gpu_from_host(g)
                         if g.dtype == 'float32' else g for g in grad)
            args = [_cupy_array_to_theano_input(x) for x in args]

        grad_func = theano.function(
            inputs=self._inputs + gs,
            outputs=grad,
            on_unused_input='ignore')

        outs = grad_func(*args)
        assert len(outs) == len(inputs)

        if gpu:
            device = theano.sandbox.cuda.active_device_number()
            outs = [_theano_output_to_cupy_array(x, device) for x in outs]

        outputs = []
        for o, i in zip(outs, inputs):
            if i.dtype.kind != 'f':
                o = None
            elif o.dtype != i.dtype:
                o = o.astype(i.dtype)
            outputs.append(o)
        return tuple(outputs)


def _cupy_array_to_theano_input(x):
    # CudaNdarray only supports float32
    if isinstance(x, cuda.cupy.ndarray) and x.dtype == numpy.float32:
        return _cupy_array_to_theano_array(x)
    else:
        return cuda.to_cpu(x)


def _cupy_array_to_theano_array(x):
    if six.PY2:
        ptr = long(x.data.ptr)  # NOQA
    else:
        ptr = int(x.data.ptr)
    # CuPy's stride is written in bytes, but CudaNdarray uses size
    strides = [s // 4 for s in x.strides]
    return theano.sandbox.cuda.from_gpu_pointer(ptr, x.shape, strides, x)


class CudaNdarrayMemory(object):

    def __init__(self, array, device):
        self._array = array
        self.device = cuda.Device(device)
        self.ptr = array.gpudata


def _theano_array_to_cupy_array(x, device):
    mem = CudaNdarrayMemory(x, device)
    memptr = cuda.cupy.cuda.MemoryPointer(mem, 0)
    # Theano's CudaNdarray is always float32
    return cuda.cupy.ndarray(x.shape, dtype=numpy.float32, memptr=memptr)


def _theano_output_to_cupy_array(x, device):
    if x is None:
        return None
    elif isinstance(x, theano.sandbox.cuda.CudaNdarray):
        return _theano_array_to_cupy_array(x, device)
    else:
        return cuda.to_gpu(x)
