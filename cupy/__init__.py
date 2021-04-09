import functools as _functools
import sys as _sys
import warnings as _warnings

import numpy as _numpy

from cupy import _environment
from cupy import _version


_environment._setup_win32_dll_directory()  # NOQA
_environment._preload_libraries()  # NOQA


try:
    from cupy import _core  # NOQA
except ImportError as e:
    # _core is a c-extension module.
    # When a user cannot import _core, it represents that CuPy is not correctly
    # built.
    _exc_info = _sys.exc_info()
    _msg = ('''\
CuPy is not correctly installed.

If you are using wheel distribution (cupy-cudaXX), make sure that the version of CuPy you installed matches with the version of CUDA on your host.
Also, confirm that only one CuPy package is installed:
  $ pip freeze

If you are building CuPy from source, please check your environment, uninstall CuPy and reinstall it with:
  $ pip install cupy --no-cache-dir -vvvv

Check the Installation Guide for details:
  https://docs.cupy.dev/en/latest/install.html

original error: {}'''.format(_exc_info[1]))  # NOQA

    raise ImportError(_msg) from e


from cupy import cuda  # NOQA
# Do not make `cupy.cupyx` available because it is confusing.
import cupyx as _cupyx  # NOQA


def is_available():
    return cuda.is_available()


__version__ = _version.__version__


import cupy._core.fusion  # NOQA
from cupy import fft  # NOQA
from cupy import linalg  # NOQA
from cupy import polynomial  # NOQA
from cupy import random  # NOQA
# `cupy.sparse` is deprecated in v8
from cupy import sparse  # NOQA
from cupy import testing  # NOQA  # NOQA


# import class and function
from cupy._core import ndarray  # NOQA
from cupy._core import ufunc  # NOQA


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
# https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
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

# -----------------------------------------------------------------------------
# Built-in Python types
# -----------------------------------------------------------------------------

# =============================================================================
# Routines
#
# The order of these declarations are borrowed from the NumPy document:
# https://docs.scipy.org/doc/numpy/reference/routines.html
# =============================================================================

# -----------------------------------------------------------------------------
# Array creation routines
# -----------------------------------------------------------------------------
from cupy._creation.basic import empty  # NOQA
from cupy._creation.basic import empty_like  # NOQA
from cupy._creation.basic import eye  # NOQA
from cupy._creation.basic import full  # NOQA
from cupy._creation.basic import full_like  # NOQA
from cupy._creation.basic import identity  # NOQA
from cupy._creation.basic import ones  # NOQA
from cupy._creation.basic import ones_like  # NOQA
from cupy._creation.basic import zeros  # NOQA
from cupy._creation.basic import zeros_like  # NOQA

from cupy._creation.from_data import copy  # NOQA
from cupy._creation.from_data import array  # NOQA
from cupy._creation.from_data import asanyarray  # NOQA
from cupy._creation.from_data import asarray  # NOQA
from cupy._creation.from_data import ascontiguousarray  # NOQA
from cupy._creation.from_data import fromfile  # NOQA

from cupy._creation.ranges import arange  # NOQA
from cupy._creation.ranges import linspace  # NOQA
from cupy._creation.ranges import logspace  # NOQA
from cupy._creation.ranges import meshgrid  # NOQA
from cupy._creation.ranges import mgrid  # NOQA
from cupy._creation.ranges import ogrid  # NOQA

from cupy._creation.matrix import diag  # NOQA
from cupy._creation.matrix import diagflat  # NOQA
from cupy._creation.matrix import tri  # NOQA
from cupy._creation.matrix import tril  # NOQA
from cupy._creation.matrix import triu  # NOQA

# -----------------------------------------------------------------------------
# Functional routines
# -----------------------------------------------------------------------------
from cupy._functional.piecewise import piecewise  # NOQA
from cupy._functional.vectorize import vectorize  # NOQA
from cupy.lib._shape_base import apply_along_axis  # NOQA

# -----------------------------------------------------------------------------
# Array manipulation routines
# -----------------------------------------------------------------------------
from cupy._manipulation.basic import copyto  # NOQA

from cupy._manipulation.shape import shape  # NOQA
from cupy._manipulation.shape import ravel  # NOQA
from cupy._manipulation.shape import reshape  # NOQA

from cupy._manipulation.transpose import moveaxis  # NOQA
from cupy._manipulation.transpose import rollaxis  # NOQA
from cupy._manipulation.transpose import swapaxes  # NOQA
from cupy._manipulation.transpose import transpose  # NOQA

from cupy._manipulation.dims import atleast_1d  # NOQA
from cupy._manipulation.dims import atleast_2d  # NOQA
from cupy._manipulation.dims import atleast_3d  # NOQA
from cupy._manipulation.dims import broadcast  # NOQA
from cupy._manipulation.dims import broadcast_arrays  # NOQA
from cupy._manipulation.dims import broadcast_to  # NOQA
from cupy._manipulation.dims import expand_dims  # NOQA
from cupy._manipulation.dims import squeeze  # NOQA

from cupy._manipulation.join import column_stack  # NOQA
from cupy._manipulation.join import concatenate  # NOQA
from cupy._manipulation.join import dstack  # NOQA
from cupy._manipulation.join import hstack  # NOQA
from cupy._manipulation.join import stack  # NOQA
from cupy._manipulation.join import vstack  # NOQA

from cupy._manipulation.kind import asfortranarray  # NOQA
from cupy._manipulation.kind import require  # NOQA

from cupy._manipulation.split import array_split  # NOQA
from cupy._manipulation.split import dsplit  # NOQA
from cupy._manipulation.split import hsplit  # NOQA
from cupy._manipulation.split import split  # NOQA
from cupy._manipulation.split import vsplit  # NOQA

from cupy._manipulation.tiling import repeat  # NOQA
from cupy._manipulation.tiling import tile  # NOQA

from cupy._manipulation.add_remove import append  # NOQA
from cupy._manipulation.add_remove import resize  # NOQA
from cupy._manipulation.add_remove import unique  # NOQA
from cupy._manipulation.add_remove import trim_zeros  # NOQA

from cupy._manipulation.rearrange import flip  # NOQA
from cupy._manipulation.rearrange import fliplr  # NOQA
from cupy._manipulation.rearrange import flipud  # NOQA
from cupy._manipulation.rearrange import roll  # NOQA
from cupy._manipulation.rearrange import rot90  # NOQA

# -----------------------------------------------------------------------------
# Binary operations
# -----------------------------------------------------------------------------
from cupy._binary.elementwise import bitwise_and  # NOQA
from cupy._binary.elementwise import bitwise_or  # NOQA
from cupy._binary.elementwise import bitwise_xor  # NOQA
from cupy._binary.elementwise import bitwise_not  # NOQA
from cupy._binary.elementwise import invert  # NOQA
from cupy._binary.elementwise import left_shift  # NOQA
from cupy._binary.elementwise import right_shift  # NOQA

from cupy._binary.packing import packbits  # NOQA
from cupy._binary.packing import unpackbits  # NOQA


def binary_repr(num, width=None):
    """Return the binary representation of the input number as a string.

    .. seealso:: :func:`numpy.binary_repr`
    """
    return _numpy.binary_repr(num, width)


# -----------------------------------------------------------------------------
# Data type routines (borrowed from NumPy)
# -----------------------------------------------------------------------------
def can_cast(from_, to, casting='safe'):
    """Returns True if cast between data types can occur according to the
    casting rule. If from is a scalar or array scalar, also returns True if the
    scalar value can be cast without overflow or truncation to an integer.

    .. seealso:: :func:`numpy.can_cast`
    """
    from_ = from_.dtype if isinstance(from_, cupy.ndarray) else from_
    return _numpy.can_cast(from_, to, casting=casting)


def common_type(*arrays):
    """Return a scalar type which is common to the input arrays.

    .. seealso:: :func:`numpy.common_type`
    """
    if len(arrays) == 0:
        return _numpy.float16

    default_float_dtype = _numpy.dtype('float64')
    dtypes = []
    for a in arrays:
        if a.dtype.kind == 'b':
            raise TypeError('can\'t get common type for non-numeric array')
        elif a.dtype.kind in 'iu':
            dtypes.append(default_float_dtype)
        else:
            dtypes.append(a.dtype)

    return _functools.reduce(_numpy.promote_types, dtypes).type


def result_type(*arrays_and_dtypes):
    """Returns the type that results from applying the NumPy type promotion
    rules to the arguments.

    .. seealso:: :func:`numpy.result_type`
    """
    dtypes = [a.dtype if isinstance(a, cupy.ndarray)
              else a for a in arrays_and_dtypes]
    return _numpy.result_type(*dtypes)


from numpy import min_scalar_type  # NOQA
from numpy import obj2sctype  # NOQA
from numpy import promote_types  # NOQA

from numpy import dtype  # NOQA
from numpy import format_parser  # NOQA

from numpy import finfo  # NOQA
from numpy import iinfo  # NOQA
from numpy import MachAr  # NOQA

from numpy import find_common_type  # NOQA
from numpy import issctype  # NOQA
from numpy import issubclass_  # NOQA
from numpy import issubdtype  # NOQA
from numpy import issubsctype  # NOQA

from numpy import mintypecode  # NOQA
from numpy import sctype2char  # NOQA
from numpy import typename  # NOQA

# -----------------------------------------------------------------------------
# Optionally Scipy-accelerated routines
# -----------------------------------------------------------------------------
# TODO(beam2d): Implement it

# -----------------------------------------------------------------------------
# Discrete Fourier Transform
# -----------------------------------------------------------------------------
# TODO(beam2d): Implement it

# -----------------------------------------------------------------------------
# Indexing routines
# -----------------------------------------------------------------------------
from cupy._indexing.generate import c_  # NOQA
from cupy._indexing.generate import indices  # NOQA
from cupy._indexing.generate import ix_  # NOQA
from cupy._indexing.generate import r_  # NOQA
from cupy._indexing.generate import ravel_multi_index  # NOQA
from cupy._indexing.generate import unravel_index  # NOQA

from cupy._indexing.indexing import choose  # NOQA
from cupy._indexing.indexing import compress  # NOQA
from cupy._indexing.indexing import diagonal  # NOQA
from cupy._indexing.indexing import extract  # NOQA
from cupy._indexing.indexing import select  # NOQA
from cupy._indexing.indexing import take  # NOQA
from cupy._indexing.indexing import take_along_axis  # NOQA

from cupy._indexing.insert import place  # NOQA
from cupy._indexing.insert import put  # NOQA
from cupy._indexing.insert import putmask  # NOQA
from cupy._indexing.insert import fill_diagonal  # NOQA
from cupy._indexing.insert import diag_indices  # NOQA
from cupy._indexing.insert import diag_indices_from  # NOQA

from cupy._indexing.iterate import flatiter  # NOQA

# -----------------------------------------------------------------------------
# Input and output
# -----------------------------------------------------------------------------
from cupy._io.npz import load  # NOQA
from cupy._io.npz import save  # NOQA
from cupy._io.npz import savez  # NOQA
from cupy._io.npz import savez_compressed  # NOQA

from cupy._io.formatting import array_repr  # NOQA
from cupy._io.formatting import array_str  # NOQA


def base_repr(number, base=2, padding=0):  # NOQA (needed to avoid redefinition of `number`)
    """Return a string representation of a number in the given base system.

    .. seealso:: :func:`numpy.base_repr`
    """
    return _numpy.base_repr(number, base, padding)


# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------
from cupy.linalg._einsum import einsum  # NOQA

from cupy.linalg._product import cross  # NOQA
from cupy.linalg._product import dot  # NOQA
from cupy.linalg._product import inner  # NOQA
from cupy.linalg._product import kron  # NOQA
from cupy.linalg._product import matmul  # NOQA
from cupy.linalg._product import outer  # NOQA
from cupy.linalg._product import tensordot  # NOQA
from cupy.linalg._product import vdot  # NOQA

from cupy.linalg._norms import trace  # NOQA

# -----------------------------------------------------------------------------
# Logic functions
# -----------------------------------------------------------------------------
from cupy._logic.comparison import allclose  # NOQA
from cupy._logic.comparison import array_equal  # NOQA
from cupy._logic.comparison import isclose  # NOQA

from cupy._logic.content import isfinite  # NOQA
from cupy._logic.content import isinf  # NOQA
from cupy._logic.content import isnan  # NOQA

from cupy._logic.truth import in1d  # NOQA
from cupy._logic.truth import isin  # NOQA

from cupy._logic.type_test import iscomplex  # NOQA
from cupy._logic.type_test import iscomplexobj  # NOQA
from cupy._logic.type_test import isfortran  # NOQA
from cupy._logic.type_test import isreal  # NOQA
from cupy._logic.type_test import isrealobj  # NOQA

from cupy._logic.truth import in1d  # NOQA
from cupy._logic.truth import isin  # NOQA


def isscalar(element):
    """Returns True if the type of num is a scalar type.

    .. seealso:: :func:`numpy.isscalar`
    """
    return _numpy.isscalar(element)


from cupy._logic.ops import logical_and  # NOQA
from cupy._logic.ops import logical_not  # NOQA
from cupy._logic.ops import logical_or  # NOQA
from cupy._logic.ops import logical_xor  # NOQA

from cupy._logic.comparison import equal  # NOQA
from cupy._logic.comparison import greater  # NOQA
from cupy._logic.comparison import greater_equal  # NOQA
from cupy._logic.comparison import less  # NOQA
from cupy._logic.comparison import less_equal  # NOQA
from cupy._logic.comparison import not_equal  # NOQA

from cupy._logic.truth import all  # NOQA
from cupy._logic.truth import any  # NOQA

# ------------------------------------------------------------------------------
# Polynomial functions
# ------------------------------------------------------------------------------
from cupy.lib._polynomial import poly1d  # NOQA
from cupy.lib._routines_poly import polyadd  # NOQA
from cupy.lib._routines_poly import polysub  # NOQA
from cupy.lib._routines_poly import polymul  # NOQA
from cupy.lib._routines_poly import polyfit  # NOQA
from cupy.lib._routines_poly import polyval  # NOQA
from cupy.lib._routines_poly import roots  # NOQA

# -----------------------------------------------------------------------------
# Mathematical functions
# -----------------------------------------------------------------------------
from cupy._math.trigonometric import arccos  # NOQA
from cupy._math.trigonometric import arcsin  # NOQA
from cupy._math.trigonometric import arctan  # NOQA
from cupy._math.trigonometric import arctan2  # NOQA
from cupy._math.trigonometric import cos  # NOQA
from cupy._math.trigonometric import deg2rad  # NOQA
from cupy._math.trigonometric import degrees  # NOQA
from cupy._math.trigonometric import hypot  # NOQA
from cupy._math.trigonometric import rad2deg  # NOQA
from cupy._math.trigonometric import radians  # NOQA
from cupy._math.trigonometric import sin  # NOQA
from cupy._math.trigonometric import tan  # NOQA
from cupy._math.trigonometric import unwrap  # NOQA

from cupy._math.hyperbolic import arccosh  # NOQA
from cupy._math.hyperbolic import arcsinh  # NOQA
from cupy._math.hyperbolic import arctanh  # NOQA
from cupy._math.hyperbolic import cosh  # NOQA
from cupy._math.hyperbolic import sinh  # NOQA
from cupy._math.hyperbolic import tanh  # NOQA

from cupy._math.rounding import around  # NOQA
from cupy._math.rounding import ceil  # NOQA
from cupy._math.rounding import fix  # NOQA
from cupy._math.rounding import floor  # NOQA
from cupy._math.rounding import rint  # NOQA
from cupy._math.rounding import round_  # NOQA
from cupy._math.rounding import round_ as round  # NOQA
from cupy._math.rounding import trunc  # NOQA

from cupy._math.sumprod import prod  # NOQA
from cupy._math.sumprod import sum  # NOQA
from cupy._math.sumprod import cumprod  # NOQA
from cupy._math.sumprod import cumsum  # NOQA
from cupy._math.sumprod import nancumprod  # NOQA
from cupy._math.sumprod import nancumsum  # NOQA
from cupy._math.sumprod import nansum  # NOQA
from cupy._math.sumprod import nanprod  # NOQA
from cupy._math.sumprod import diff  # NOQA
from cupy._math.sumprod import gradient  # NOQA
from cupy._math.window import bartlett  # NOQA
from cupy._math.window import blackman  # NOQA
from cupy._math.window import hamming  # NOQA
from cupy._math.window import hanning  # NOQA
from cupy._math.window import kaiser  # NOQA

from cupy._math.explog import exp  # NOQA
from cupy._math.explog import exp2  # NOQA
from cupy._math.explog import expm1  # NOQA
from cupy._math.explog import log  # NOQA
from cupy._math.explog import log10  # NOQA
from cupy._math.explog import log1p  # NOQA
from cupy._math.explog import log2  # NOQA
from cupy._math.explog import logaddexp  # NOQA
from cupy._math.explog import logaddexp2  # NOQA

from cupy._math.special import i0  # NOQA
from cupy._math.special import sinc  # NOQA

from cupy._math.floating import copysign  # NOQA
from cupy._math.floating import frexp  # NOQA
from cupy._math.floating import ldexp  # NOQA
from cupy._math.floating import nextafter  # NOQA
from cupy._math.floating import signbit  # NOQA

from cupy._math.rational import gcd  # NOQA
from cupy._math.rational import lcm  # NOQA

from cupy._math.arithmetic import add  # NOQA
from cupy._math.arithmetic import divide  # NOQA
from cupy._math.arithmetic import divmod  # NOQA
from cupy._math.arithmetic import floor_divide  # NOQA
from cupy._math.arithmetic import fmod  # NOQA
from cupy._math.arithmetic import modf  # NOQA
from cupy._math.arithmetic import multiply  # NOQA
from cupy._math.arithmetic import negative  # NOQA
from cupy._math.arithmetic import power  # NOQA
from cupy._math.arithmetic import reciprocal  # NOQA
from cupy._math.arithmetic import remainder  # NOQA
from cupy._math.arithmetic import remainder as mod  # NOQA
from cupy._math.arithmetic import subtract  # NOQA
from cupy._math.arithmetic import true_divide  # NOQA

from cupy._math.arithmetic import angle  # NOQA
from cupy._math.arithmetic import conjugate as conj  # NOQA
from cupy._math.arithmetic import conjugate  # NOQA
from cupy._math.arithmetic import imag  # NOQA
from cupy._math.arithmetic import real  # NOQA

from cupy._math.misc import absolute as abs  # NOQA
from cupy._math.misc import absolute  # NOQA
from cupy._math.misc import cbrt  # NOQA
from cupy._math.misc import clip  # NOQA
from cupy._math.misc import fmax  # NOQA
from cupy._math.misc import fmin  # NOQA
from cupy._math.misc import interp  # NOQA
from cupy._math.misc import maximum  # NOQA
from cupy._math.misc import minimum  # NOQA
from cupy._math.misc import nan_to_num  # NOQA
from cupy._math.misc import sign  # NOQA
from cupy._math.misc import sqrt  # NOQA
from cupy._math.misc import square  # NOQA
from cupy._math.misc import convolve  # NOQA

# -----------------------------------------------------------------------------
# Miscellaneous routines
# -----------------------------------------------------------------------------
from cupy._misc.memory_ranges import may_share_memory  # NOQA
from cupy._misc.memory_ranges import shares_memory  # NOQA
from cupy._misc.who import who  # NOQA


# -----------------------------------------------------------------------------
# Padding
# -----------------------------------------------------------------------------
from cupy._padding.pad import pad  # NOQA


# -----------------------------------------------------------------------------
# Sorting, searching, and counting
# -----------------------------------------------------------------------------
from cupy._sorting.count import count_nonzero  # NOQA

from cupy._sorting.search import argmax  # NOQA
from cupy._sorting.search import argmin  # NOQA
from cupy._sorting.search import argwhere  # NOQA
from cupy._sorting.search import flatnonzero  # NOQA
from cupy._sorting.search import nanargmax  # NOQA
from cupy._sorting.search import nanargmin  # NOQA
from cupy._sorting.search import nonzero  # NOQA
from cupy._sorting.search import searchsorted  # NOQA
from cupy._sorting.search import where  # NOQA

from cupy._sorting.sort import argpartition  # NOQA
from cupy._sorting.sort import argsort  # NOQA
from cupy._sorting.sort import lexsort  # NOQA
from cupy._sorting.sort import msort  # NOQA
from cupy._sorting.sort import sort_complex  # NOQA
from cupy._sorting.sort import partition  # NOQA
from cupy._sorting.sort import sort  # NOQA

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
from cupy._statistics.correlation import corrcoef  # NOQA
from cupy._statistics.correlation import cov  # NOQA
from cupy._statistics.correlation import correlate  # NOQA

from cupy._statistics.order import amax  # NOQA
from cupy._statistics.order import amax as max  # NOQA
from cupy._statistics.order import amin  # NOQA
from cupy._statistics.order import amin as min  # NOQA
from cupy._statistics.order import nanmax  # NOQA
from cupy._statistics.order import nanmin  # NOQA
from cupy._statistics.order import percentile  # NOQA
from cupy._statistics.order import ptp  # NOQA
from cupy._statistics.order import quantile  # NOQA

from cupy._statistics.meanvar import median  # NOQA
from cupy._statistics.meanvar import average  # NOQA
from cupy._statistics.meanvar import mean  # NOQA
from cupy._statistics.meanvar import std  # NOQA
from cupy._statistics.meanvar import var  # NOQA
from cupy._statistics.meanvar import nanmedian  # NOQA
from cupy._statistics.meanvar import nanmean  # NOQA
from cupy._statistics.meanvar import nanstd  # NOQA
from cupy._statistics.meanvar import nanvar  # NOQA

from cupy._statistics.histogram import bincount  # NOQA
from cupy._statistics.histogram import digitize  # NOQA
from cupy._statistics.histogram import histogram  # NOQA
from cupy._statistics.histogram import histogram2d  # NOQA
from cupy._statistics.histogram import histogramdd  # NOQA

# -----------------------------------------------------------------------------
# Undocumented functions
# -----------------------------------------------------------------------------
from cupy._core import size  # NOQA


def ndim(a):
    """Returns the number of dimensions of an array.

    Args:
        a (array-like): If it is not already an `cupy.ndarray`, a conversion
            via :func:`numpy.asarray` is attempted.

    Returns:
        (int): The number of dimensions in `a`.

    """
    try:
        return a.ndim
    except AttributeError:
        return _numpy.ndim(a)


# -----------------------------------------------------------------------------
# CuPy specific functions
# -----------------------------------------------------------------------------

from cupy._util import clear_memo  # NOQA
from cupy._util import memoize  # NOQA

from cupy._core import ElementwiseKernel  # NOQA
from cupy._core import RawKernel  # NOQA
from cupy._core import RawModule  # NOQA
from cupy._core._reduction import ReductionKernel  # NOQA

# -----------------------------------------------------------------------------
# DLPack
# -----------------------------------------------------------------------------

from cupy._core import fromDlpack  # NOQA


def asnumpy(a, stream=None, order='C'):
    """Returns an array on the host memory from an arbitrary source array.

    Args:
        a: Arbitrary object that can be converted to :class:`numpy.ndarray`.
        stream (cupy.cuda.Stream): CUDA stream object. If it is specified, then
            the device-to-host copy runs asynchronously. Otherwise, the copy is
            synchronous. Note that if ``a`` is not a :class:`cupy.ndarray`
            object, then this argument has no effect.
        order ({'C', 'F', 'A'}): The desired memory layout of the host
            array. When ``order`` is 'A', it uses 'F' if ``a`` is
            fortran-contiguous and 'C' otherwise.
    Returns:
        numpy.ndarray: Converted array on the host memory.

    """
    if isinstance(a, ndarray):
        return a.get(stream=stream, order=order)
    elif hasattr(a, "__cuda_array_interface__"):
        return array(a).get(stream=stream, order=order)
    else:
        return _numpy.asarray(a, order=order)


_cupy = _sys.modules[__name__]


def get_array_module(*args):
    """Returns the array module for arguments.

    This function is used to implement CPU/GPU generic code. If at least one of
    the arguments is a :class:`cupy.ndarray` object, the :mod:`cupy` module is
    returned.

    Args:
        args: Values to determine whether NumPy or CuPy should be used.

    Returns:
        module: :mod:`cupy` or :mod:`numpy` is returned based on the types of
        the arguments.

    .. admonition:: Example

       A NumPy/CuPy generic function can be written as follows

       >>> def softplus(x):
       ...     xp = cupy.get_array_module(x)
       ...     return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))

    """
    for arg in args:
        if isinstance(arg, (ndarray, _cupyx.scipy.sparse.spmatrix,
                            cupy._core.fusion._FusionVarArray,
                            cupy._core.new_fusion._ArrayProxy)):
            return _cupy
    return _numpy


fuse = cupy._core.fusion.fuse

disable_experimental_feature_warning = False


# set default allocator
_default_memory_pool = cuda.MemoryPool()
_default_pinned_memory_pool = cuda.PinnedMemoryPool()

cuda.set_allocator(_default_memory_pool.malloc)
cuda.set_pinned_memory_allocator(_default_pinned_memory_pool.malloc)


def get_default_memory_pool():
    """Returns CuPy default memory pool for GPU memory.

    Returns:
        cupy.cuda.MemoryPool: The memory pool object.

    .. note::
       If you want to disable memory pool, please use the following code.

       >>> cupy.cuda.set_allocator(None)

    """
    return _default_memory_pool


def get_default_pinned_memory_pool():
    """Returns CuPy default memory pool for pinned memory.

    Returns:
        cupy.cuda.PinnedMemoryPool: The memory pool object.

    .. note::
       If you want to disable memory pool, please use the following code.

       >>> cupy.cuda.set_pinned_memory_allocator(None)

    """
    return _default_pinned_memory_pool


def show_config():
    """Prints the current runtime configuration to standard output."""
    _sys.stdout.write(str(_cupyx.get_runtime_info()))
    _sys.stdout.flush()


if _sys.version_info >= (3, 7):
    _deprecated_attrs = {
        'int': (int, 'cupy.int_'),
        'bool': (bool, 'cupy.bool_'),
        'float': (float, 'cupy.float_'),
        'complex': (complex, 'cupy.complex_'),
    }

    def __getattr__(name):
        value = _deprecated_attrs.get(name)
        if value is None:
            raise AttributeError(
                f"module 'cupy' has no attribute {name!r}")
        attr, eq_attr = value
        _warnings.warn(
            f'`cupy.{name}` is a deprecated alias for the Python scalar type '
            f'`{name}`. Please use the builtin `{name}` or its corresponding '
            f'NumPy scalar type `{eq_attr}` instead.',
            DeprecationWarning, stacklevel=2
        )
        return attr
else:
    # Does not emit warnings.
    from builtins import int
    from builtins import bool
    from builtins import float
    from builtins import complex
