"""
Wrapper around cupy.ndarray
to support fallback of methods such as `ndarray.func()`
"""

import numpy as np

import cupy as cp
from cupyx.fallback_mode import utils
from cupyx.fallback_mode import data_transfer


def make_method(name):
    def method(self, *args, **kwargs):

        args = utils._convert_ndarrays(args)

        cupy_method = getattr(cp.ndarray, name)

        res = cupy_method(self._array, *args, **kwargs)

        return utils._get_fallback_ndarray(res)

    return method


def _create_magic_methods():

    _magic_methods = [

        # Comparison operators:
        '__richcmp__',

        # Truth value of an array (bool):
        '__nonzero__',

        # Unary operations:
        '__neg__', '__pos__', '__abs__', '__invert__',

        # Arithmetic:
        '__add__', '__sub__', '__mul__', '__matmul__', '__div__',
        '__truediv__', '__floordiv__', '__mod__', '__divmod__', '__pow__',
        '__lshift__', '__rshift__', '__and__', '__or__', '__xor__',

        # Arithmetic, in-place:
        '__iadd__', '__isub__', '__imul__', '__idiv__', '__itruediv__',
        '__ifloordiv__', '__imod__', '__ipow__', '__ilshift__', '__irshift__',
        '__iand__', '__ior__', '__ixor__',

        # For standard library functions:
        '__copy__', '__deepcopy__', '__reduce__',

        # Container customization:
        '__iter__', '__len__', '__getitem__', '__setitem__',

        # Conversion:
        '__int__', '__float__', '__complex__', '__oct__', '__hex__',

        # String representations:
        '__repr__', '__str__'
    ]

    for method in _magic_methods:
        setattr(ndarray, method, make_method(method))


class ndarray:

    def __init__(self, array):
        self._array = array
        self.func = None

    def __getattr__(self, attr):

        self.func = getattr(cp.ndarray, attr, None)
        if not callable(self.func) and self.func is not None:
            return getattr(self._array, attr)

        if self.func is not None:
            return self._call_cupy_ndarray

        self.func = getattr(np.ndarray, attr)
        return self._call_numpy_ndarray

    def _call_cupy_ndarray(self, *args, **kwargs):

        res = self.func(self._array, *args, **kwargs)

        return utils._get_fallback_ndarray(res)

    def _call_numpy_ndarray(self, *args, **kwargs):

        numpy_args, numpy_kwargs = data_transfer._get_numpy_args(args, kwargs)

        numpy_array = cp.asnumpy(self._array)
        numpy_res = self.func(numpy_array, *numpy_args, **numpy_kwargs)

        cupy_res = data_transfer._get_cupy_result(numpy_res)

        return utils._get_fallback_ndarray(cupy_res)
