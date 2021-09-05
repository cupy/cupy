from __future__ import annotations

from ._dtypes import (
    _boolean_dtypes,
    _floating_dtypes,
    _integer_dtypes,
    _integer_or_boolean_dtypes,
    _numeric_dtypes,
    _result_type,
)
from ._array_object import Array

import cupy as cp


def abs(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.abs <cupy.abs>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in abs")
    return Array._new(cp.abs(x._array))


# Note: the function name is different here
def acos(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.arccos <cupy.arccos>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in acos")
    return Array._new(cp.arccos(x._array))


# Note: the function name is different here
def acosh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.arccosh <cupy.arccosh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in acosh")
    return Array._new(cp.arccosh(x._array))


def add(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.add <cupy.add>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in add")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.add(x1._array, x2._array))


# Note: the function name is different here
def asin(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.arcsin <cupy.arcsin>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in asin")
    return Array._new(cp.arcsin(x._array))


# Note: the function name is different here
def asinh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.arcsinh <cupy.arcsinh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in asinh")
    return Array._new(cp.arcsinh(x._array))


# Note: the function name is different here
def atan(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.arctan <cupy.arctan>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atan")
    return Array._new(cp.arctan(x._array))


# Note: the function name is different here
def atan2(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.arctan2 <cupy.arctan2>`.

    See its docstring for more information.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atan2")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.arctan2(x1._array, x2._array))


# Note: the function name is different here
def atanh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.arctanh <cupy.arctanh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atanh")
    return Array._new(cp.arctanh(x._array))


def bitwise_and(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.bitwise_and <cupy.bitwise_and>`.

    See its docstring for more information.
    """
    if (
        x1.dtype not in _integer_or_boolean_dtypes
        or x2.dtype not in _integer_or_boolean_dtypes
    ):
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_and")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.bitwise_and(x1._array, x2._array))


# Note: the function name is different here
def bitwise_left_shift(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.left_shift <cupy.left_shift>`.

    See its docstring for more information.
    """
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError("Only integer dtypes are allowed in bitwise_left_shift")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    # Note: bitwise_left_shift is only defined for x2 nonnegative.
    if cp.any(x2._array < 0):
        raise ValueError("bitwise_left_shift(x1, x2) is only defined for x2 >= 0")
    return Array._new(cp.left_shift(x1._array, x2._array))


# Note: the function name is different here
def bitwise_invert(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.invert <cupy.invert>`.

    See its docstring for more information.
    """
    if x.dtype not in _integer_or_boolean_dtypes:
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_invert")
    return Array._new(cp.invert(x._array))


def bitwise_or(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.bitwise_or <cupy.bitwise_or>`.

    See its docstring for more information.
    """
    if (
        x1.dtype not in _integer_or_boolean_dtypes
        or x2.dtype not in _integer_or_boolean_dtypes
    ):
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_or")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.bitwise_or(x1._array, x2._array))


# Note: the function name is different here
def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.right_shift <cupy.right_shift>`.

    See its docstring for more information.
    """
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError("Only integer dtypes are allowed in bitwise_right_shift")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    # Note: bitwise_right_shift is only defined for x2 nonnegative.
    if cp.any(x2._array < 0):
        raise ValueError("bitwise_right_shift(x1, x2) is only defined for x2 >= 0")
    return Array._new(cp.right_shift(x1._array, x2._array))


def bitwise_xor(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.bitwise_xor <cupy.bitwise_xor>`.

    See its docstring for more information.
    """
    if (
        x1.dtype not in _integer_or_boolean_dtypes
        or x2.dtype not in _integer_or_boolean_dtypes
    ):
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_xor")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.bitwise_xor(x1._array, x2._array))


def ceil(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.ceil <cupy.ceil>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in ceil")
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of ceil is the same as the input
        return x
    return Array._new(cp.ceil(x._array))


def cos(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.cos <cupy.cos>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in cos")
    return Array._new(cp.cos(x._array))


def cosh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.cosh <cupy.cosh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in cosh")
    return Array._new(cp.cosh(x._array))


def divide(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.divide <cupy.divide>`.

    See its docstring for more information.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in divide")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.divide(x1._array, x2._array))


def equal(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.equal <cupy.equal>`.

    See its docstring for more information.
    """
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.equal(x1._array, x2._array))


def exp(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.exp <cupy.exp>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in exp")
    return Array._new(cp.exp(x._array))


def expm1(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.expm1 <cupy.expm1>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in expm1")
    return Array._new(cp.expm1(x._array))


def floor(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.floor <cupy.floor>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in floor")
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of floor is the same as the input
        return x
    return Array._new(cp.floor(x._array))


def floor_divide(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.floor_divide <cupy.floor_divide>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in floor_divide")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.floor_divide(x1._array, x2._array))


def greater(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.greater <cupy.greater>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in greater")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.greater(x1._array, x2._array))


def greater_equal(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.greater_equal <cupy.greater_equal>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in greater_equal")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.greater_equal(x1._array, x2._array))


def isfinite(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.isfinite <cupy.isfinite>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isfinite")
    return Array._new(cp.isfinite(x._array))


def isinf(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.isinf <cupy.isinf>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isinf")
    return Array._new(cp.isinf(x._array))


def isnan(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.isnan <cupy.isnan>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isnan")
    return Array._new(cp.isnan(x._array))


def less(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.less <cupy.less>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in less")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.less(x1._array, x2._array))


def less_equal(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.less_equal <cupy.less_equal>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in less_equal")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.less_equal(x1._array, x2._array))


def log(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.log <cupy.log>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log")
    return Array._new(cp.log(x._array))


def log1p(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.log1p <cupy.log1p>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log1p")
    return Array._new(cp.log1p(x._array))


def log2(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.log2 <cupy.log2>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log2")
    return Array._new(cp.log2(x._array))


def log10(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.log10 <cupy.log10>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log10")
    return Array._new(cp.log10(x._array))


def logaddexp(x1: Array, x2: Array) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.logaddexp <cupy.logaddexp>`.

    See its docstring for more information.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in logaddexp")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.logaddexp(x1._array, x2._array))


def logical_and(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.logical_and <cupy.logical_and>`.

    See its docstring for more information.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_and")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.logical_and(x1._array, x2._array))


def logical_not(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.logical_not <cupy.logical_not>`.

    See its docstring for more information.
    """
    if x.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_not")
    return Array._new(cp.logical_not(x._array))


def logical_or(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.logical_or <cupy.logical_or>`.

    See its docstring for more information.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_or")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.logical_or(x1._array, x2._array))


def logical_xor(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.logical_xor <cupy.logical_xor>`.

    See its docstring for more information.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_xor")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.logical_xor(x1._array, x2._array))


def multiply(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.multiply <cupy.multiply>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in multiply")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.multiply(x1._array, x2._array))


def negative(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.negative <cupy.negative>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in negative")
    return Array._new(cp.negative(x._array))


def not_equal(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.not_equal <cupy.not_equal>`.

    See its docstring for more information.
    """
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.not_equal(x1._array, x2._array))


def positive(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.positive <cupy.positive>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in positive")
    return Array._new(cp.positive(x._array))


# Note: the function name is different here
def pow(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.power <cupy.power>`.

    See its docstring for more information.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in pow")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.power(x1._array, x2._array))


def remainder(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.remainder <cupy.remainder>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in remainder")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.remainder(x1._array, x2._array))


def round(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.round <cupy.round>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in round")
    return Array._new(cp.round(x._array))


def sign(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.sign <cupy.sign>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sign")
    return Array._new(cp.sign(x._array))


def sin(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.sin <cupy.sin>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sin")
    return Array._new(cp.sin(x._array))


def sinh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.sinh <cupy.sinh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sinh")
    return Array._new(cp.sinh(x._array))


def square(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.square <cupy.square>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in square")
    return Array._new(cp.square(x._array))


def sqrt(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.sqrt <cupy.sqrt>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sqrt")
    return Array._new(cp.sqrt(x._array))


def subtract(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.subtract <cupy.subtract>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in subtract")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(cp.subtract(x1._array, x2._array))


def tan(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.tan <cupy.tan>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in tan")
    return Array._new(cp.tan(x._array))


def tanh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.tanh <cupy.tanh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in tanh")
    return Array._new(cp.tanh(x._array))


def trunc(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`cp.trunc <cupy.trunc>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in trunc")
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of trunc is the same as the input
        return x
    return Array._new(cp.trunc(x._array))
