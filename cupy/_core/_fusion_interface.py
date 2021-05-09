import numpy

from cupy._core._dtype import get_dtype
import cupy
from cupy._core import _fusion_thread_local
from cupy._core import core
from cupy._core._scalar import get_typename


_thread_local = _fusion_thread_local.thread_local


_dtype_to_astype_dict = None


def _set_dtype_to_astype_dict():
    """Set a dict with dtypes and astype ufuncs to `_dtype_to_astype_dict`.

    Creates a ufunc for type cast operations, and set a dict with keys
    as the dtype of the output array and values as astype ufuncs.
    This function is called at most once.
    """
    global _dtype_to_astype_dict
    _dtype_to_astype_dict = {}

    dtype_list = [numpy.dtype(type_char) for type_char in '?bhilqBHILQefdFD']

    for t in dtype_list:
        name = 'astype_{}'.format(t)
        rules = tuple(['{}->{}'.format(s.char, t.char) for s in dtype_list])
        command = 'out0 = static_cast< {} >(in0)'.format(get_typename(t))
        _dtype_to_astype_dict[t] = core.create_ufunc(name, rules, command)


class _VariableProxy:
    """Abstracted array/scalar object passed to the target function.
    """

    def __init__(self, content):
        assert isinstance(content, cupy._core._fusion_variable._TraceVariable)
        self.content = content

    def __neg__(self):
        return cupy.negative(self)

    def __add__(self, other):
        return cupy.add(self, other)

    def __radd__(self, other):
        return cupy.add(other, self)

    def __sub__(self, other):
        return cupy.subtract(self, other)

    def __rsub__(self, other):
        return cupy.subtract(other, self)

    def __mul__(self, other):
        return cupy.multiply(self, other)

    def __rmul__(self, other):
        return cupy.multiply(other, self)

    def __div__(self, other):
        return cupy.divide(self, other)

    def __rdiv__(self, other):
        return cupy.divide(other, self)

    def __truediv__(self, other):
        return cupy.true_divide(self, other)

    def __rtruediv__(self, other):
        return cupy.true_divide(other, self)

    def __floordiv__(self, other):
        return cupy.floor_divide(self, other)

    def __rfloordiv__(self, other):
        return cupy.floor_divide(other, self)

    def __mod__(self, other):
        return cupy.remainder(self, other)

    def __rmod__(self, other):
        return cupy.remainder(other, self)

    def __pow__(self, other):
        return cupy.power(self, other)

    def __lshift__(self, other):
        return cupy.left_shift(self, other)

    def __rlshift__(self, other):
        return cupy.left_shift(other, self)

    def __rshift__(self, other):
        return cupy.right_shift(self, other)

    def __rrshift__(self, other):
        return cupy.right_shift(other, self)

    def __invert__(self):
        return cupy.invert(self)

    def __and__(self, other):
        return cupy.bitwise_and(self, other)

    def __rand__(self, other):
        return cupy.bitwise_and(other, self)

    def __or__(self, other):
        return cupy.bitwise_or(self, other)

    def __ror__(self, other):
        return cupy.bitwise_or(other, self)

    def __xor__(self, other):
        return cupy.bitwise_xor(self, other)

    def __rxor__(self, other):
        return cupy.bitwise_xor(other, self)

    def __lt__(self, other):
        return cupy.less(self, other)

    def __le__(self, other):
        return cupy.less_equal(self, other)

    def __eq__(self, other):
        return cupy.equal(self, other)

    def __ne__(self, other):
        return cupy.not_equal(self, other)

    def __ge__(self, other):
        return cupy.greater_equal(self, other)

    def __gt__(self, other):
        return cupy.greater(self, other)

    def copy(self):
        return cupy.copy(self)

    def astype(self, dtype, order=None, casting=None, subok=None, copy=True):
        dtype = get_dtype(dtype)
        if order is not None:
            raise TypeError('order is not supported yet')
        if casting is not None:
            raise TypeError('casting is not supported yet')
        if subok is not None:
            raise TypeError('subok is not supported yet')
        if not copy and self.dtype == dtype:
            return self
        if _dtype_to_astype_dict is None:
            _set_dtype_to_astype_dict()
        return _dtype_to_astype_dict[dtype](self)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        return cupy.sum(
            self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):
        return cupy.prod(
            self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def max(self, axis=None, out=None, keepdims=False):
        return cupy.max(self, axis=axis, out=out, keepdims=keepdims)

    def min(self, axis=None, out=None, keepdims=False):
        return cupy.min(self, axis=axis, out=out, keepdims=keepdims)

    def all(self, axis=None, out=None, keepdims=False):
        return cupy.all(self, axis=axis, out=out, keepdims=keepdims)

    def any(self, axis=None, out=None, keepdims=False):
        return cupy.any(self, axis=axis, out=out, keepdims=keepdims)

    @property
    def dtype(self):
        return self.content.dtype

    @property
    def ndim(self):
        return self.content.ndim

    @property
    def shape(self):
        raise NotImplementedError('`shape` is not supported, currently.')


class _ScalarProxy(_VariableProxy):
    """An abstracted scalar object passed to the target function.

    Attributes:
        dtype(dtype): The dtype of the array.
        imag(_ArrayProxy): The imaginary part of the array (Not implemented)
        real(_ArrayProxy): The real part of the array (Not implemented)
        ndim(int): The number of dimensions of the array.
    """

    def __repr__(self):
        return '_ScalarProxy({}, dtype={})'.format(
            self._emit_param_name(), self.dtype)


class _ArrayProxy(_VariableProxy):
    """An abstracted array object passed to the target function.

    Attributes:
        dtype(dtype): The dtype of the array.
        imag(_ArrayProxy): The imaginary part of the array (Not implemented)
        real(_ArrayProxy): The real part of the array (Not implemented)
        ndim(int): The number of dimensions of the array.
    """

    def __repr__(self):
        return '_ArrayProxy([...], dtype=\'{}\', ndim={})'.format(
            self.dtype.char, self.ndim)

    def _inplace_op(self, ufunc, other):
        return ufunc(self, other, self)

    def __iadd__(self, other):
        return self._inplace_op(cupy.add, other)

    def __isub__(self, other):
        return self._inplace_op(cupy.subtract, other)

    def __imul__(self, other):
        return self._inplace_op(cupy.multiply, other)

    def __idiv__(self, other):
        return self._inplace_op(cupy.divide, other)

    def __itruediv__(self, other):
        return self._inplace_op(cupy.true_divide, other)

    def __ifloordiv__(self, other):
        return self._inplace_op(cupy.floor_divide, other)

    def __imod__(self, other):
        return self._inplace_op(cupy.remainder, other)

    def __ipow__(self, other):
        return self._inplace_op(cupy.power, other)

    def __ilshift__(self, other):
        return self._inplace_op(cupy.left_shift, other)

    def __irshift__(self, other):
        return self._inplace_op(cupy.right_shift, other)

    def __iand__(self, other):
        return self._inplace_op(cupy.bitwise_and, other)

    def __ior__(self, other):
        return self._inplace_op(cupy.bitwise_or, other)

    def __ixor__(self, other):
        return self._inplace_op(cupy.bitwise_xor, other)

    def __getitem__(self, index):
        return _fusion_thread_local.call_indexing(self, index)

    def __setitem__(self, slices, value):
        if slices is Ellipsis or (
                isinstance(slices, slice) and slices == slice(None)):
            _fusion_thread_local.call_ufunc(
                core.elementwise_copy, value, out=self)
        else:
            raise ValueError('The fusion supports `[...]` or `[:]`.')
