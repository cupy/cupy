"""
From NumPy
==========

APIs listed here will be re-exported under `cupy.*` namespace.

This file must not depend on any other CuPy modules.
"""

import warnings as _warnings

import numpy as _numpy


# =============================================================================
# Constants (borrowed from NumPy)
# =============================================================================
from numpy import e  # NOQA
from numpy import euler_gamma  # NOQA
from numpy import Inf  # NOQA
from numpy import inf  # NOQA
from numpy import Infinity  # NOQA
from numpy import infty  # NOQA
from numpy import NAN  # NOQA
from numpy import NaN  # NOQA
from numpy import nan  # NOQA
from numpy import newaxis  # == None  # NOQA
from numpy import NINF  # NOQA
from numpy import NZERO  # NOQA
from numpy import pi  # NOQA
from numpy import PINF  # NOQA
from numpy import PZERO  # NOQA


# =============================================================================
# Data types (borrowed from NumPy)
#
# The order of these declarations are borrowed from the NumPy document:
# https://numpy.org/doc/stable/reference/arrays.scalars.html
# =============================================================================

# -----------------------------------------------------------------------------
# Generic types
# -----------------------------------------------------------------------------
from numpy import complexfloating  # NOQA
from numpy import floating  # NOQA
from numpy import generic  # NOQA
from numpy import inexact  # NOQA
from numpy import integer  # NOQA
from numpy import number  # NOQA
from numpy import signedinteger  # NOQA
from numpy import unsignedinteger  # NOQA

# Not supported by CuPy:
# from numpy import flexible
# from numpy import character

# -----------------------------------------------------------------------------
# Booleans
# -----------------------------------------------------------------------------
from numpy import bool_  # NOQA
from numpy import bool8  # NOQA

# -----------------------------------------------------------------------------
# Integers
# -----------------------------------------------------------------------------
from numpy import byte  # NOQA
from numpy import short  # NOQA
from numpy import intc  # NOQA
from numpy import int_  # NOQA
from numpy import longlong  # NOQA
from numpy import intp  # NOQA
from numpy import int0  # NOQA
from numpy import int8  # NOQA
from numpy import int16  # NOQA
from numpy import int32  # NOQA
from numpy import int64  # NOQA

# -----------------------------------------------------------------------------
# Unsigned integers
# -----------------------------------------------------------------------------
from numpy import ubyte  # NOQA
from numpy import ushort  # NOQA
from numpy import uintc  # NOQA
from numpy import uint  # NOQA
from numpy import ulonglong  # NOQA
from numpy import uintp  # NOQA
from numpy import uint0  # NOQA
from numpy import uint8  # NOQA
from numpy import uint16  # NOQA
from numpy import uint32  # NOQA
from numpy import uint64  # NOQA

# -----------------------------------------------------------------------------
# Floating-point numbers
# -----------------------------------------------------------------------------
from numpy import half  # NOQA
from numpy import single  # NOQA
from numpy import double  # NOQA
from numpy import float_  # NOQA
from numpy import longfloat  # NOQA
from numpy import float16  # NOQA
from numpy import float32  # NOQA
from numpy import float64  # NOQA

# Not supported by CuPy:
# from numpy import float96
# from numpy import float128

# -----------------------------------------------------------------------------
# Complex floating-point numbers
# -----------------------------------------------------------------------------
from numpy import csingle  # NOQA
from numpy import singlecomplex  # NOQA
from numpy import cdouble  # NOQA
from numpy import cfloat  # NOQA
from numpy import complex_  # NOQA
from numpy import complex64  # NOQA
from numpy import complex128  # NOQA

# Not supported by CuPy:
# from numpy import complex192
# from numpy import complex256
# from numpy import clongfloat

# -----------------------------------------------------------------------------
# Any Python object
# -----------------------------------------------------------------------------

# Not supported by CuPy:
# from numpy import object_
# from numpy import bytes_
# from numpy import unicode_
# from numpy import void


# =============================================================================
# Deprecated NumPy APIs
# =============================================================================

_deprecated_apis = [
    'MachAr',  # NumPy 1.22
]

_deprecated_scalar_aliases = {  # NumPy 1.20
    'int': (int, 'cupy.int_'),
    'bool': (bool, 'cupy.bool_'),
    'float': (float, 'cupy.float_'),
    'complex': (complex, 'cupy.complex_'),
}


def __getattr__(name):
    value = _deprecated_scalar_aliases.get(name)
    if value is not None:
        attr, eq_attr = value
        _warnings.warn(
            f'`cupy.{name}` is a deprecated alias for the Python scalar type '
            f'`{name}`. Please use the builtin `{name}` or its corresponding '
            f'NumPy scalar type `{eq_attr}` instead.',
            DeprecationWarning, stacklevel=2
        )
        return attr

    if name in _deprecated_apis:
        return getattr(_numpy, name)

    raise AttributeError(f"module 'cupy' has no attribute {name!r}")
