"""
class ndarray is wrapper around cupy.ndarray
to support fallback of methods of type `ndarray.func()`
"""
import numpy as np

import cupy as cp
from cupyx.fallback_mode import data_transfer

try:
    from cupyx.fallback_mode import fallback
except ImportError:
    import sys
    fallback = sys.modules[__package__ + '.fallback']


class ndarray:
    """
    Wrapper around cupy.ndarray
    Gets initialized with a cupy ndarray.
    """

    def __init__(self, array):
        self._array = array

    def __getattr__(self, attr):
        """
        Catches attributes corresponding to ndarray.

        Args:
            attr (str): Attribute of ndarray class.

        Returns:
            (_RecursiveAttr object, self._array.attr):
            Returns_RecursiveAttr object with numpy_object, cupy_object.
            Returns self._array.attr if attr is not callable.
        """

        cupy_object = getattr(cp.ndarray, attr, None)

        numpy_object = getattr(np.ndarray, attr)

        if not callable(numpy_object):
            return getattr(self._array, attr)

        return fallback._RecursiveAttr(numpy_object, cupy_object, self)

    def _get_array(self):
        """
        Returns _array (cupy.ndarray) of ndarray object.
        """
        return self._array


def _get_cupy_ndarray(args, kwargs):
    return data_transfer._get_xp_args(
        ndarray, ndarray._get_array, (args, kwargs))


def _get_fallback_ndarray(cupy_res):
    return data_transfer._get_xp_args(cp.ndarray, ndarray, cupy_res)


# Decorator for ndarray magic methods
def make_method(name):
    def method(self, *args, **kwargs):

        args, kwargs = _get_cupy_ndarray(args, kwargs)

        cupy_method = getattr(cp.ndarray, name)

        res = cupy_method(self._array, *args, **kwargs)

        return _get_fallback_ndarray(res)

    return method


def _create_magic_methods():
    """
    Set magic methods of cupy.ndarray as methods of utils.ndarray.
    """

    _magic_methods = [

        # Comparison operators:
        '__eq__', '__ne__', '__lt__', '__gt__', '__le__', '__ge__',

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


_create_magic_methods()
