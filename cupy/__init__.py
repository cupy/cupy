import functools as _functools
import sys as _sys

import numpy as _numpy

from cupy import _environment
from cupy import _version


_environment._detect_duplicate_installation()
_environment._setup_win32_dll_directory()
_environment._preload_library('cutensor')


try:
    from cupy import _core
except ImportError as exc:
    raise ImportError(f'''
================================================================
{_environment._diagnose_import_error()}

Original error:
  {type(exc).__name__}: {exc}
================================================================
''') from exc


from cupy import cuda
# Do not make `cupy.cupyx` available because it is confusing.
import cupyx as _cupyx


def is_available():
    return cuda.is_available()


__version__ = _version.__version__


from cupy import fft
from cupy import linalg
from cupy import polynomial
from cupy import random
# `cupy.sparse` is deprecated in v8
from cupy import sparse
from cupy import testing


# import class and function
from cupy._core import ndarray
from cupy._core import ufunc


# =============================================================================
# Constants (borrowed from NumPy)
# =============================================================================
from numpy import e
from numpy import euler_gamma
from numpy import inf
from numpy import nan
from numpy import newaxis  # == None
from numpy import pi

# APIs to be removed in NumPy 2.0.
# Remove these when bumping the baseline API to NumPy 2.0.
# https://github.com/cupy/cupy/pull/7800
PINF = Inf = Infinity = infty = inf
NINF = -inf
NAN = NaN = nan
PZERO = 0.0
NZERO = -0.0

# =============================================================================
# Data types (borrowed from NumPy)
#
# The order of these declarations are borrowed from the NumPy document:
# https://numpy.org/doc/stable/reference/arrays.scalars.html
# =============================================================================

# -----------------------------------------------------------------------------
# Generic types
# -----------------------------------------------------------------------------
from numpy import complexfloating
from numpy import floating
from numpy import generic
from numpy import inexact
from numpy import integer
from numpy import number
from numpy import signedinteger
from numpy import unsignedinteger

# Not supported by CuPy:
# from numpy import flexible
# from numpy import character

# -----------------------------------------------------------------------------
# Booleans
# -----------------------------------------------------------------------------
from numpy import bool_

# -----------------------------------------------------------------------------
# Integers
# -----------------------------------------------------------------------------
from numpy import byte
from numpy import short
from numpy import intc
from numpy import int_
from numpy import longlong
from numpy import intp
from numpy import int8
from numpy import int16
from numpy import int32
from numpy import int64

# -----------------------------------------------------------------------------
# Unsigned integers
# -----------------------------------------------------------------------------
from numpy import ubyte
from numpy import ushort
from numpy import uintc
from numpy import uint
from numpy import ulonglong
from numpy import uintp
from numpy import uint8
from numpy import uint16
from numpy import uint32
from numpy import uint64

# -----------------------------------------------------------------------------
# Floating-point numbers
# -----------------------------------------------------------------------------
from numpy import half
from numpy import single
from numpy import double
from numpy import float64 as float_
# from numpy import longfloat  # XXX
from numpy import float16
from numpy import float32
from numpy import float64

# Not supported by CuPy:
# from numpy import float96
# from numpy import float128

# -----------------------------------------------------------------------------
# Complex floating-point numbers
# -----------------------------------------------------------------------------
from numpy import csingle
from numpy import complex64 as singlecomplex
from numpy import cdouble
from numpy import complex128 as cfloat
from numpy import complex128 as complex_
from numpy import complex64
from numpy import complex128

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
# https://numpy.org/doc/stable/reference/routines.html
# =============================================================================

# -----------------------------------------------------------------------------
# Array creation routines
# -----------------------------------------------------------------------------
from cupy._creation.basic import empty
from cupy._creation.basic import empty_like
from cupy._creation.basic import eye
from cupy._creation.basic import full
from cupy._creation.basic import full_like
from cupy._creation.basic import identity
from cupy._creation.basic import ones
from cupy._creation.basic import ones_like
from cupy._creation.basic import zeros
from cupy._creation.basic import zeros_like

from cupy._creation.from_data import copy
from cupy._creation.from_data import array
from cupy._creation.from_data import asanyarray
from cupy._creation.from_data import asarray
from cupy._creation.from_data import ascontiguousarray
from cupy._creation.from_data import fromfile
from cupy._creation.from_data import fromfunction
from cupy._creation.from_data import fromiter
from cupy._creation.from_data import frombuffer
from cupy._creation.from_data import fromstring
from cupy._creation.from_data import loadtxt
from cupy._creation.from_data import genfromtxt

from cupy._creation.ranges import arange
from cupy._creation.ranges import linspace
from cupy._creation.ranges import logspace
from cupy._creation.ranges import meshgrid
from cupy._creation.ranges import mgrid
from cupy._creation.ranges import ogrid

from cupy._creation.matrix import diag
from cupy._creation.matrix import diagflat
from cupy._creation.matrix import tri
from cupy._creation.matrix import tril
from cupy._creation.matrix import triu
from cupy._creation.matrix import vander

# -----------------------------------------------------------------------------
# Functional routines
# -----------------------------------------------------------------------------
from cupy._functional.piecewise import piecewise
from cupy._functional.vectorize import vectorize
from cupy.lib._shape_base import apply_along_axis
from cupy.lib._shape_base import put_along_axis

# -----------------------------------------------------------------------------
# Array manipulation routines
# -----------------------------------------------------------------------------
from cupy._manipulation.basic import copyto

from cupy._manipulation.shape import shape
from cupy._manipulation.shape import ravel
from cupy._manipulation.shape import reshape

from cupy._manipulation.transpose import moveaxis
from cupy._manipulation.transpose import rollaxis
from cupy._manipulation.transpose import swapaxes
from cupy._manipulation.transpose import transpose

from cupy._manipulation.dims import atleast_1d
from cupy._manipulation.dims import atleast_2d
from cupy._manipulation.dims import atleast_3d
from cupy._manipulation.dims import broadcast
from cupy._manipulation.dims import broadcast_arrays
from cupy._manipulation.dims import broadcast_to
from cupy._manipulation.dims import expand_dims
from cupy._manipulation.dims import squeeze

from cupy._manipulation.join import column_stack
from cupy._manipulation.join import concatenate
from cupy._manipulation.join import dstack
from cupy._manipulation.join import hstack
from cupy._manipulation.join import stack
from cupy._manipulation.join import vstack
from cupy._manipulation.join import vstack as row_stack

from cupy._manipulation.kind import asarray_chkfinite
from cupy._manipulation.kind import asfarray
from cupy._manipulation.kind import asfortranarray
from cupy._manipulation.kind import require

from cupy._manipulation.split import array_split
from cupy._manipulation.split import dsplit
from cupy._manipulation.split import hsplit
from cupy._manipulation.split import split
from cupy._manipulation.split import vsplit

from cupy._manipulation.tiling import repeat
from cupy._manipulation.tiling import tile

from cupy._manipulation.add_remove import delete
from cupy._manipulation.add_remove import append
from cupy._manipulation.add_remove import resize
from cupy._manipulation.add_remove import unique
from cupy._manipulation.add_remove import trim_zeros

from cupy._manipulation.rearrange import flip
from cupy._manipulation.rearrange import fliplr
from cupy._manipulation.rearrange import flipud
from cupy._manipulation.rearrange import roll
from cupy._manipulation.rearrange import rot90

# Borrowed from NumPy
if hasattr(_numpy, 'broadcast_shapes'):  # NumPy 1.20
    from numpy import broadcast_shapes

# -----------------------------------------------------------------------------
# Binary operations
# -----------------------------------------------------------------------------
from cupy._binary.elementwise import bitwise_and
from cupy._binary.elementwise import bitwise_or
from cupy._binary.elementwise import bitwise_xor
from cupy._binary.elementwise import bitwise_not
from cupy._binary.elementwise import invert
from cupy._binary.elementwise import left_shift
from cupy._binary.elementwise import right_shift

from cupy._binary.packing import packbits
from cupy._binary.packing import unpackbits


def binary_repr(num, width=None):
    """Return the binary representation of the input number as a string.

    .. seealso:: :func:`numpy.binary_repr`
    """
    return _numpy.binary_repr(num, width)


# -----------------------------------------------------------------------------
# Data type routines (mostly borrowed from NumPy)
# -----------------------------------------------------------------------------
def can_cast(from_, to, casting='safe'):
    """Returns True if cast between data types can occur according to the
    casting rule. If from is a scalar or array scalar, also returns True if the
    scalar value can be cast without overflow or truncation to an integer.

    .. seealso:: :func:`numpy.can_cast`
    """
    from_ = from_.dtype if isinstance(from_, ndarray) else from_
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
    dtypes = [a.dtype if isinstance(a, ndarray)
              else a for a in arrays_and_dtypes]
    return _numpy.result_type(*dtypes)


from cupy._core.core import min_scalar_type

from numpy import promote_types

from numpy import dtype

from numpy import finfo
from numpy import iinfo

from numpy import issubdtype

from numpy import mintypecode
from numpy import typename

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
from cupy._indexing.generate import c_
from cupy._indexing.generate import indices
from cupy._indexing.generate import ix_
from cupy._indexing.generate import mask_indices
from cupy._indexing.generate import tril_indices
from cupy._indexing.generate import tril_indices_from
from cupy._indexing.generate import triu_indices
from cupy._indexing.generate import triu_indices_from
from cupy._indexing.generate import r_
from cupy._indexing.generate import ravel_multi_index
from cupy._indexing.generate import unravel_index

from cupy._indexing.indexing import choose
from cupy._indexing.indexing import compress
from cupy._indexing.indexing import diagonal
from cupy._indexing.indexing import extract
from cupy._indexing.indexing import select
from cupy._indexing.indexing import take
from cupy._indexing.indexing import take_along_axis

from cupy._indexing.insert import place
from cupy._indexing.insert import put
from cupy._indexing.insert import putmask
from cupy._indexing.insert import fill_diagonal
from cupy._indexing.insert import diag_indices
from cupy._indexing.insert import diag_indices_from

from cupy._indexing.iterate import flatiter

# Borrowed from NumPy
from numpy import index_exp
from numpy import ndindex
from numpy import s_

# -----------------------------------------------------------------------------
# Input and output
# -----------------------------------------------------------------------------
from cupy._io.npz import load
from cupy._io.npz import save
from cupy._io.npz import savez
from cupy._io.npz import savez_compressed

from cupy._io.formatting import array_repr
from cupy._io.formatting import array_str
from cupy._io.formatting import array2string
from cupy._io.formatting import format_float_positional
from cupy._io.formatting import format_float_scientific

from cupy._io.text import savetxt


def base_repr(number, base=2, padding=0):  # NOQA: F811 (needed to avoid redefinition of `number`)
    """Return a string representation of a number in the given base system.

    .. seealso:: :func:`numpy.base_repr`
    """
    return _numpy.base_repr(number, base, padding)


# Borrowed from NumPy
from numpy import get_printoptions
from numpy import set_printoptions
from numpy import printoptions


# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------
from cupy.linalg._einsum import einsum

from cupy.linalg._product import cross
from cupy.linalg._product import dot
from cupy.linalg._product import inner
from cupy.linalg._product import kron
from cupy.linalg._product import matmul
from cupy.linalg._product import outer
from cupy.linalg._product import tensordot
from cupy.linalg._product import vdot

from cupy.linalg._norms import trace

# -----------------------------------------------------------------------------
# Logic functions
# -----------------------------------------------------------------------------
from cupy._logic.comparison import allclose
from cupy._logic.comparison import array_equal
from cupy._logic.comparison import array_equiv
from cupy._logic.comparison import isclose

from cupy._logic.content import isfinite
from cupy._logic.content import isinf
from cupy._logic.content import isnan
from cupy._logic.content import isneginf
from cupy._logic.content import isposinf

from cupy._logic.type_testing import iscomplex
from cupy._logic.type_testing import iscomplexobj
from cupy._logic.type_testing import isfortran
from cupy._logic.type_testing import isreal
from cupy._logic.type_testing import isrealobj

from cupy._logic.truth import in1d
from cupy._logic.truth import intersect1d
from cupy._logic.truth import isin
from cupy._logic.truth import setdiff1d
from cupy._logic.truth import setxor1d
from cupy._logic.truth import union1d


def isscalar(element):
    """Returns True if the type of num is a scalar type.

    .. seealso:: :func:`numpy.isscalar`
    """
    return _numpy.isscalar(element)


from cupy._logic.ops import logical_and
from cupy._logic.ops import logical_not
from cupy._logic.ops import logical_or
from cupy._logic.ops import logical_xor

from cupy._logic.comparison import equal
from cupy._logic.comparison import greater
from cupy._logic.comparison import greater_equal
from cupy._logic.comparison import less
from cupy._logic.comparison import less_equal
from cupy._logic.comparison import not_equal

from cupy._logic.truth import all
from cupy._logic.truth import alltrue
from cupy._logic.truth import any
from cupy._logic.truth import sometrue

# ------------------------------------------------------------------------------
# Polynomial functions
# ------------------------------------------------------------------------------
from cupy.lib._polynomial import poly1d
from cupy.lib._routines_poly import poly
from cupy.lib._routines_poly import polyadd
from cupy.lib._routines_poly import polysub
from cupy.lib._routines_poly import polymul
from cupy.lib._routines_poly import polyfit
from cupy.lib._routines_poly import polyval
from cupy.lib._routines_poly import roots

# Borrowed from NumPy
from cupy.exceptions import RankWarning

# -----------------------------------------------------------------------------
# Mathematical functions
# -----------------------------------------------------------------------------
from cupy._math.trigonometric import arccos
from cupy._math.trigonometric import arcsin
from cupy._math.trigonometric import arctan
from cupy._math.trigonometric import arctan2
from cupy._math.trigonometric import cos
from cupy._math.trigonometric import deg2rad
from cupy._math.trigonometric import degrees
from cupy._math.trigonometric import hypot
from cupy._math.trigonometric import rad2deg
from cupy._math.trigonometric import radians
from cupy._math.trigonometric import sin
from cupy._math.trigonometric import tan
from cupy._math.trigonometric import unwrap

from cupy._math.hyperbolic import arccosh
from cupy._math.hyperbolic import arcsinh
from cupy._math.hyperbolic import arctanh
from cupy._math.hyperbolic import cosh
from cupy._math.hyperbolic import sinh
from cupy._math.hyperbolic import tanh

from cupy._math.rounding import around
from cupy._math.rounding import ceil
from cupy._math.rounding import fix
from cupy._math.rounding import floor
from cupy._math.rounding import rint
from cupy._math.rounding import round
from cupy._math.rounding import round_
from cupy._math.rounding import trunc

from cupy._math.sumprod import prod
from cupy._math.sumprod import product
from cupy._math.sumprod import sum
from cupy._math.sumprod import cumprod
from cupy._math.sumprod import cumproduct
from cupy._math.sumprod import cumsum
from cupy._math.sumprod import ediff1d
from cupy._math.sumprod import nancumprod
from cupy._math.sumprod import nancumsum
from cupy._math.sumprod import nansum
from cupy._math.sumprod import nanprod
from cupy._math.sumprod import diff
from cupy._math.sumprod import gradient
from cupy._math.sumprod import trapz
from cupy._math.window import bartlett
from cupy._math.window import blackman
from cupy._math.window import hamming
from cupy._math.window import hanning
from cupy._math.window import kaiser

from cupy._math.explog import exp
from cupy._math.explog import exp2
from cupy._math.explog import expm1
from cupy._math.explog import log
from cupy._math.explog import log10
from cupy._math.explog import log1p
from cupy._math.explog import log2
from cupy._math.explog import logaddexp
from cupy._math.explog import logaddexp2

from cupy._math.special import i0
from cupy._math.special import sinc

from cupy._math.floating import copysign
from cupy._math.floating import frexp
from cupy._math.floating import ldexp
from cupy._math.floating import nextafter
from cupy._math.floating import signbit

from cupy._math.rational import gcd
from cupy._math.rational import lcm

from cupy._math.arithmetic import add
from cupy._math.arithmetic import divide
from cupy._math.arithmetic import divmod
from cupy._math.arithmetic import floor_divide
from cupy._math.arithmetic import float_power
from cupy._math.arithmetic import fmod
from cupy._math.arithmetic import modf
from cupy._math.arithmetic import multiply
from cupy._math.arithmetic import negative
from cupy._math.arithmetic import positive
from cupy._math.arithmetic import power
from cupy._math.arithmetic import reciprocal
from cupy._math.arithmetic import remainder
from cupy._math.arithmetic import remainder as mod
from cupy._math.arithmetic import subtract
from cupy._math.arithmetic import true_divide

from cupy._math.arithmetic import angle
from cupy._math.arithmetic import conjugate as conj
from cupy._math.arithmetic import conjugate
from cupy._math.arithmetic import imag
from cupy._math.arithmetic import real

from cupy._math.misc import absolute as abs
from cupy._math.misc import absolute
from cupy._math.misc import cbrt
from cupy._math.misc import clip
from cupy._math.misc import fabs
from cupy._math.misc import fmax
from cupy._math.misc import fmin
from cupy._math.misc import interp
from cupy._math.misc import maximum
from cupy._math.misc import minimum
from cupy._math.misc import nan_to_num
from cupy._math.misc import real_if_close
from cupy._math.misc import sign
from cupy._math.misc import heaviside
from cupy._math.misc import sqrt
from cupy._math.misc import square
from cupy._math.misc import convolve

# -----------------------------------------------------------------------------
# Miscellaneous routines
# -----------------------------------------------------------------------------
from cupy._misc.byte_bounds import byte_bounds
from cupy._misc.memory_ranges import may_share_memory
from cupy._misc.memory_ranges import shares_memory
from cupy._misc.who import who

# Borrowed from NumPy
from numpy import iterable
from cupy.exceptions import AxisError


# -----------------------------------------------------------------------------
# Padding
# -----------------------------------------------------------------------------
from cupy._padding.pad import pad


# -----------------------------------------------------------------------------
# Sorting, searching, and counting
# -----------------------------------------------------------------------------
from cupy._sorting.count import count_nonzero

from cupy._sorting.search import argmax
from cupy._sorting.search import argmin
from cupy._sorting.search import argwhere
from cupy._sorting.search import flatnonzero
from cupy._sorting.search import nanargmax
from cupy._sorting.search import nanargmin
from cupy._sorting.search import nonzero
from cupy._sorting.search import searchsorted
from cupy._sorting.search import where

from cupy._sorting.sort import argpartition
from cupy._sorting.sort import argsort
from cupy._sorting.sort import lexsort
from cupy._sorting.sort import msort
from cupy._sorting.sort import sort_complex
from cupy._sorting.sort import partition
from cupy._sorting.sort import sort

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
from cupy._statistics.correlation import corrcoef
from cupy._statistics.correlation import cov
from cupy._statistics.correlation import correlate

from cupy._statistics.order import amax
from cupy._statistics.order import amax as max
from cupy._statistics.order import amin
from cupy._statistics.order import amin as min
from cupy._statistics.order import nanmax
from cupy._statistics.order import nanmin
from cupy._statistics.order import percentile
from cupy._statistics.order import ptp
from cupy._statistics.order import quantile

from cupy._statistics.meanvar import median
from cupy._statistics.meanvar import average
from cupy._statistics.meanvar import mean
from cupy._statistics.meanvar import std
from cupy._statistics.meanvar import var
from cupy._statistics.meanvar import nanmedian
from cupy._statistics.meanvar import nanmean
from cupy._statistics.meanvar import nanstd
from cupy._statistics.meanvar import nanvar

from cupy._statistics.histogram import bincount
from cupy._statistics.histogram import digitize
from cupy._statistics.histogram import histogram
from cupy._statistics.histogram import histogram2d
from cupy._statistics.histogram import histogramdd

# -----------------------------------------------------------------------------
# Classes without their own docs
# -----------------------------------------------------------------------------
from cupy.exceptions import ComplexWarning
from cupy.exceptions import ModuleDeprecationWarning
from cupy.exceptions import TooHardError
from cupy.exceptions import VisibleDeprecationWarning


# -----------------------------------------------------------------------------
# Undocumented functions
# -----------------------------------------------------------------------------
from cupy._core import size


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

from cupy._util import clear_memo
from cupy._util import memoize

from cupy._core import ElementwiseKernel
from cupy._core import RawKernel
from cupy._core import RawModule
from cupy._core._reduction import ReductionKernel

# -----------------------------------------------------------------------------
# DLPack
# -----------------------------------------------------------------------------

from cupy._core import fromDlpack
from cupy._core import from_dlpack


def asnumpy(a, stream=None, order='C', out=None, *, blocking=True):
    """Returns an array on the host memory from an arbitrary source array.

    Args:
        a: Arbitrary object that can be converted to :class:`numpy.ndarray`.
        stream (cupy.cuda.Stream): CUDA stream object. If given, the
            stream is used to perform the copy. Otherwise, the current
            stream is used. Note that if ``a`` is not a :class:`cupy.ndarray`
            object, then this argument has no effect.
        order ({'C', 'F', 'A'}): The desired memory layout of the host
            array. When ``order`` is 'A', it uses 'F' if the array is
            fortran-contiguous and 'C' otherwise. The ``order`` will be
            ignored if ``out`` is specified.
        out (numpy.ndarray): The output array to be written to. It must have
            compatible shape and dtype with those of ``a``'s.
        blocking (bool): If set to ``False``, the copy runs asynchronously
            on the given (if given) or current stream, and users are
            responsible for ensuring the stream order. Default is ``True``,
            so the copy is synchronous (with respect to the host).

    Returns:
        numpy.ndarray: Converted array on the host memory.

    """
    if isinstance(a, ndarray):
        return a.get(stream=stream, order=order, out=out, blocking=blocking)
    elif hasattr(a, "__cuda_array_interface__"):
        return array(a).get(
            stream=stream, order=order, out=out, blocking=blocking)
    else:
        temp = _numpy.asarray(a, order=order)
        if out is not None:
            out[...] = temp
        else:
            out = temp
        return out


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
                            _core.fusion._FusionVarArray,
                            _core.new_fusion._ArrayProxy)):
            return _cupy
    return _numpy


fuse = _core.fusion.fuse

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


def show_config(*, _full=False):
    """Prints the current runtime configuration to standard output."""
    _sys.stdout.write(str(_cupyx.get_runtime_info(full=_full)))
    _sys.stdout.flush()


_deprecated_apis = [
    'int0',
    'uint0',
    'bool8',
]


# np 2.0: XXX shims for things removed in np 2.0

# https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/numerictypes.py#L283-L322
def issubclass_(arg1, arg2):
    try:
        return issubclass(arg1, arg2)
    except TypeError:
        return False

# https://github.com/numpy/numpy/blob/v1.26.0/numpy/core/numerictypes.py#L229-L280


def obj2sctype(rep, default=None):
    """
    Return the scalar dtype or NumPy equivalent of Python type of an object.

    Parameters
    ----------
    rep : any
        The object of which the type is returned.
    default : any, optional
        If given, this is returned for objects whose types can not be
        determined. If not given, None is returned for those objects.

    Returns
    -------
    dtype : dtype or Python type
        The data type of `rep`.

    """
    # prevent abstract classes being upcast
    if isinstance(rep, type) and issubclass(rep, _numpy.generic):
        return rep
    # extract dtype from arrays
    if isinstance(rep, _numpy.ndarray):
        return rep.dtype.type
    # fall back on dtype to convert
    try:
        res = _numpy.dtype(rep)
    except Exception:
        return default
    else:
        return res.type


# https://github.com/numpy/numpy/blob/v1.26.0/numpy/core/numerictypes.py#L326C1-L355C1
def issubsctype(arg1, arg2):
    """
    Determine if the first argument is a subclass of the second argument.

    Parameters
    ----------
    arg1, arg2 : dtype or dtype specifier
        Data-types.

    Returns
    -------
    out : bool
        The result.

    """
    return issubclass(obj2sctype(arg1), obj2sctype(arg2))


# https://github.com/numpy/numpy/blob/v1.26.0/numpy/core/numerictypes.py#L457
def sctype2char(sctype):
    """
    Return the string representation of a scalar dtype.

    Parameters
    ----------
    sctype : scalar dtype or object
        If a scalar dtype, the corresponding string character is
        returned. If an object, `sctype2char` tries to infer its scalar type
        and then return the corresponding string character.

    Returns
    -------
    typechar : str
        The string character corresponding to the scalar type.

    Raises
    ------
    ValueError
        If `sctype` is an object for which the type can not be inferred.

    """
    sctype = obj2sctype(sctype)
    if sctype is None:
        raise ValueError("unrecognized type")
    return _numpy.dtype(sctype).char


# https://github.com/numpy/numpy/blob/v1.26.0/numpy/core/numerictypes.py#L184
def issctype(rep):
    """
    Determines whether the given object represents a scalar data-type.

    Parameters
    ----------
    rep : any
        If `rep` is an instance of a scalar dtype, True is returned. If not,
        False is returned.

    Returns
    -------
    out : bool
        Boolean result of check whether `rep` is a scalar dtype.

    """
    if not isinstance(rep, (type, _numpy.dtype)):
        return False
    try:
        res = obj2sctype(rep)
        if res and res != _numpy.object_:
            return True
        return False
    except Exception:
        return False


# np 2.0: XXX shims for things moved in np 2.0
if _numpy.__version__ < "2":
    from numpy import format_parser
    from numpy import DataSource
else:
    from numpy.rec import format_parser   # type: ignore [no-redef]
    from numpy.lib.npyio import DataSource


# np 2.0: XXX shims for things removed without replacement
if _numpy.__version__ < "2":
    from numpy import find_common_type
    from numpy import set_string_function
    from numpy import get_array_wrap
    from numpy import disp
    from numpy import safe_eval
else:

    _template = '''\
''This function has been removed in NumPy v2.
Use {recommendation} instead.

CuPy has been providing this function as an alias to the NumPy
implementation, so it cannot be used in environments with NumPy
v2 installed. If you rely on this function and you cannot modify
the code to use {recommendation}, please downgrade NumPy to v1.26
or earlier.
'''

    def find_common_type(*args, **kwds):
        mesg = _template.format(
            recommendation='`promote_types` or `result_type`'
        )
        raise RuntimeError(mesg)

    def set_string_function(*args, **kwds):   # type: ignore [misc]
        mesg = _template.format(recommendation='`np.set_printoptions`')
        raise RuntimeError(mesg)

    def get_array_wrap(*args, **kwds):       # type: ignore [no-redef]
        mesg = _template.format(recommendation="<no replacement>")
        raise RuntimeError(mesg)

    def disp(*args, **kwds):   # type: ignore [misc]
        mesg = _template.format(recommendation="your own print function")
        raise RuntimeError(mesg)

    def safe_eval(*args, **kwds):  # type: ignore [misc]
        mesg = _template.format(recommendation="`ast.literal_eval`")
        raise RuntimeError(mesg)


def __getattr__(name):
    if name in _deprecated_apis:
        return getattr(_numpy, name)

    raise AttributeError(f"module 'cupy' has no attribute {name!r}")


def _embed_signatures(dirs):
    for name, value in dirs.items():
        if isinstance(value, ufunc):
            from cupy._core._kernel import _ufunc_doc_signature_formatter
            value.__doc__ = (
                _ufunc_doc_signature_formatter(value, name) +
                '\n\n' + value._doc
            )


_embed_signatures(globals())
_embed_signatures(fft.__dict__)
_embed_signatures(linalg.__dict__)
_embed_signatures(random.__dict__)
_embed_signatures(sparse.__dict__)
_embed_signatures(testing.__dict__)
