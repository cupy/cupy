"""
Utilities needed for fallback_mode.

class ndarray is wrapper around cupy.ndarray
to support fallback of methods of type `ndarray.func()`
"""

import numpy as np

import cupy as cp
from cupyx.fallback_mode import data_transfer


def _call_cupy(func, args, kwargs):
    """
    Calls cupy function with *args and **kwargs.

    Args:
        func: A cupy function that needs to be called.
        args (tuple): Arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        Result after calling func.
    """
    args, kwargs = data_transfer._get_cupy_ndarray(ndarray, args, kwargs)

    res = func(*args, **kwargs)

    return data_transfer._get_fallback_ndarray(ndarray, res)


def _call_numpy(func, args, kwargs):
    """
    Calls numpy function with *args and **kwargs.

    Args:
        func: A numpy function that needs to be called.
        args (tuple): Arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        Result after calling func.
    """
    args, kwargs = data_transfer._get_cupy_ndarray(ndarray, args, kwargs)

    numpy_args, numpy_kwargs = data_transfer._get_numpy_args(args, kwargs)

    numpy_res = func(*numpy_args, **numpy_kwargs)

    cupy_res = data_transfer._get_cupy_result(numpy_res)

    return data_transfer._get_fallback_ndarray(ndarray, cupy_res)


def make_method(name):
    def method(self, *args, **kwargs):

        args, kwargs = data_transfer._get_cupy_ndarray(ndarray, args, kwargs)

        cupy_method = getattr(cp.ndarray, name)

        res = cupy_method(self._array, *args, **kwargs)

        return data_transfer._get_fallback_ndarray(ndarray, res)

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
        self._numpy_func = False

    def __getattr__(self, attr):

        self._numpy_func = False

        self.func = getattr(cp.ndarray, attr, None)

        if not callable(self.func) and self.func is not None:
            return getattr(self._array, attr)
        if self.func is not None:
            return self.__call

        self.func = getattr(np.ndarray, attr)
        self._numpy_func = True
        return self.__call

    def __call(self, *args, **kwargs):

        args = ((self,) + args)

        if not self._numpy_func:
            return _call_cupy(self.func, args, kwargs)

        return _call_numpy(self.func, args, kwargs)

    def _get_array(self):
        return self._array


_create_magic_methods()
