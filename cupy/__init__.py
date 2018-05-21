from __future__ import division
import sys

import numpy
import six

from cupy import _version


try:
    from cupy import core  # NOQA
except ImportError:
    # core is a c-extension module.
    # When a user cannot import core, it represents that CuPy is not correctly
    # built.
    exc_info = sys.exc_info()
    msg = ('''\
CuPy is not correctly installed.

If you are using wheel distribution (cupy-cudaXX), make sure that the version of CuPy you installed matches with the version of CUDA on your host.
Also, confirm that only one CuPy package is installed:
  $ pip freeze

If you are building CuPy from source, please check your environment, uninstall CuPy and reinstall it with:
  $ pip install cupy --no-cache-dir -vvvv

Check the Installation Guide for details:
  https://docs-cupy.chainer.org/en/latest/install.html

original error: {}'''.format(exc_info[1]))  # NOQA

    six.reraise(ImportError, ImportError(msg), exc_info[2])


from cupy import cuda


def is_available():
    return cuda.is_available()


__version__ = _version.__version__


from cupy import binary  # NOQA
from cupy.core import fusion  # NOQA
from cupy import creation  # NOQA
from cupy import fft  # NOQA
from cupy import indexing  # NOQA
from cupy import io  # NOQA
from cupy import linalg  # NOQA
from cupy import manipulation  # NOQA
from cupy import padding  # NOQA
from cupy import random  # NOQA
from cupy import sorting  # NOQA
from cupy import sparse  # NOQA
from cupy import statistics  # NOQA
from cupy import testing  # NOQA  # NOQA
from cupy import util  # NOQA


# import class and function
from cupy.core import ndarray  # NOQA
from cupy.core import ufunc  # NOQA


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

from numpy import int  # NOQA

from numpy import bool  # NOQA

from numpy import float  # NOQA

from numpy import complex  # NOQA

# Not supported by CuPy:
# from numpy import object
# from numpy import unicode
# from numpy import str

# =============================================================================
# Routines
#
# The order of these declarations are borrowed from the NumPy document:
# https://docs.scipy.org/doc/numpy/reference/routines.html
# =============================================================================

# -----------------------------------------------------------------------------
# Array creation routines
# -----------------------------------------------------------------------------
from cupy.creation.basic import empty  # NOQA
from cupy.creation.basic import empty_like  # NOQA
from cupy.creation.basic import eye  # NOQA
from cupy.creation.basic import full  # NOQA
from cupy.creation.basic import full_like  # NOQA
from cupy.creation.basic import identity  # NOQA
from cupy.creation.basic import ones  # NOQA
from cupy.creation.basic import ones_like  # NOQA
from cupy.creation.basic import zeros  # NOQA
from cupy.creation.basic import zeros_like  # NOQA

from cupy.core.fusion import copy  # NOQA
from cupy.creation.from_data import array  # NOQA
from cupy.creation.from_data import asanyarray  # NOQA
from cupy.creation.from_data import asarray  # NOQA
from cupy.creation.from_data import ascontiguousarray  # NOQA

from cupy.creation.ranges import arange  # NOQA
from cupy.creation.ranges import linspace  # NOQA
from cupy.creation.ranges import logspace  # NOQA
from cupy.creation.ranges import meshgrid  # NOQA
from cupy.creation.ranges import mgrid  # NOQA
from cupy.creation.ranges import ogrid  # NOQA

from cupy.creation.matrix import diag  # NOQA
from cupy.creation.matrix import diagflat  # NOQA
from cupy.creation.matrix import tri  # NOQA
from cupy.creation.matrix import tril  # NOQA
from cupy.creation.matrix import triu  # NOQA

# -----------------------------------------------------------------------------
# Array manipulation routines
# -----------------------------------------------------------------------------
from cupy.manipulation.basic import copyto  # NOQA

from cupy.manipulation.shape import ravel  # NOQA
from cupy.manipulation.shape import reshape  # NOQA

from cupy.manipulation.transpose import moveaxis  # NOQA
from cupy.manipulation.transpose import rollaxis  # NOQA
from cupy.manipulation.transpose import swapaxes  # NOQA
from cupy.manipulation.transpose import transpose  # NOQA

from cupy.manipulation.dims import atleast_1d  # NOQA
from cupy.manipulation.dims import atleast_2d  # NOQA
from cupy.manipulation.dims import atleast_3d  # NOQA
from cupy.manipulation.dims import broadcast  # NOQA
from cupy.manipulation.dims import broadcast_arrays  # NOQA
from cupy.manipulation.dims import broadcast_to  # NOQA
from cupy.manipulation.dims import expand_dims  # NOQA
from cupy.manipulation.dims import squeeze  # NOQA

from cupy.manipulation.join import column_stack  # NOQA
from cupy.manipulation.join import concatenate  # NOQA
from cupy.manipulation.join import dstack  # NOQA
from cupy.manipulation.join import hstack  # NOQA
from cupy.manipulation.join import stack  # NOQA
from cupy.manipulation.join import vstack  # NOQA

from cupy.manipulation.kind import asfortranarray  # NOQA

from cupy.manipulation.split import array_split  # NOQA
from cupy.manipulation.split import dsplit  # NOQA
from cupy.manipulation.split import hsplit  # NOQA
from cupy.manipulation.split import split  # NOQA
from cupy.manipulation.split import vsplit  # NOQA

from cupy.manipulation.tiling import repeat  # NOQA
from cupy.manipulation.tiling import tile  # NOQA

from cupy.manipulation.add_remove import unique  # NOQA

from cupy.manipulation.rearrange import flip  # NOQA
from cupy.manipulation.rearrange import fliplr  # NOQA
from cupy.manipulation.rearrange import flipud  # NOQA
from cupy.manipulation.rearrange import roll  # NOQA
from cupy.manipulation.rearrange import rot90  # NOQA

# -----------------------------------------------------------------------------
# Binary operations
# -----------------------------------------------------------------------------
from cupy.core.fusion import bitwise_and  # NOQA
from cupy.core.fusion import bitwise_or  # NOQA
from cupy.core.fusion import bitwise_xor  # NOQA
from cupy.core.fusion import invert  # NOQA
from cupy.core.fusion import left_shift  # NOQA
from cupy.core.fusion import right_shift  # NOQA

from cupy.binary.packing import packbits  # NOQA
from cupy.binary.packing import unpackbits  # NOQA


def binary_repr(num, width=None):
    """Return the binary representation of the input number as a string.

    .. seealso:: :func:`numpy.binary_repr`
    """
    return numpy.binary_repr(num, width)


# -----------------------------------------------------------------------------
# Data type routines (borrowed from NumPy)
# -----------------------------------------------------------------------------
from numpy import can_cast  # NOQA
from numpy import common_type  # NOQA
from numpy import min_scalar_type  # NOQA
from numpy import obj2sctype  # NOQA
from numpy import promote_types  # NOQA
from numpy import result_type  # NOQA

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
from cupy.indexing.generate import c_  # NOQA
from cupy.indexing.generate import indices  # NOQA
from cupy.indexing.generate import ix_  # NOQA
from cupy.indexing.generate import r_  # NOQA
from cupy.indexing.generate import unravel_index  # NOQA

from cupy.indexing.indexing import choose  # NOQA
from cupy.indexing.indexing import diagonal  # NOQA
from cupy.indexing.indexing import take  # NOQA

from cupy.indexing.insert import fill_diagonal  # NOQA
# -----------------------------------------------------------------------------
# Input and output
# -----------------------------------------------------------------------------
from cupy.io.npz import load  # NOQA
from cupy.io.npz import save  # NOQA
from cupy.io.npz import savez  # NOQA
from cupy.io.npz import savez_compressed  # NOQA

from cupy.io.formatting import array_repr  # NOQA
from cupy.io.formatting import array_str  # NOQA


def base_repr(number, base=2, padding=0):  # NOQA (needed to avoid redefinition of `number`)
    """Return a string representation of a number in the given base system.

    .. seealso:: :func:`numpy.base_repr`
    """
    return numpy.base_repr(number, base, padding)


# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------
from cupy.linalg.einsum import einsum  # NOQA

from cupy.linalg.product import dot  # NOQA
from cupy.linalg.product import inner  # NOQA
from cupy.linalg.product import kron  # NOQA
from cupy.linalg.product import matmul  # NOQA
from cupy.linalg.product import outer  # NOQA
from cupy.linalg.product import tensordot  # NOQA
from cupy.linalg.product import vdot  # NOQA

from cupy.linalg.norms import trace  # NOQA

# -----------------------------------------------------------------------------
# Logic functions
# -----------------------------------------------------------------------------
from cupy.logic.comparison import isclose  # NOQA

from cupy.core.fusion import isfinite  # NOQA
from cupy.core.fusion import isinf  # NOQA
from cupy.core.fusion import isnan  # NOQA

from cupy.logic.type_test import iscomplex  # NOQA
from cupy.logic.type_test import iscomplexobj  # NOQA
from cupy.logic.type_test import isfortran  # NOQA
from cupy.logic.type_test import isreal  # NOQA
from cupy.logic.type_test import isrealobj  # NOQA


def isscalar(num):
    """Returns True if the type of num is a scalar type.

    .. seealso:: :func:`numpy.isscalar`
    """
    return numpy.isscalar(num)


from cupy.core.fusion import logical_and  # NOQA
from cupy.core.fusion import logical_not  # NOQA
from cupy.core.fusion import logical_or  # NOQA
from cupy.core.fusion import logical_xor  # NOQA

from cupy.core.fusion import equal  # NOQA
from cupy.core.fusion import greater  # NOQA
from cupy.core.fusion import greater_equal  # NOQA
from cupy.core.fusion import less  # NOQA
from cupy.core.fusion import less_equal  # NOQA
from cupy.core.fusion import not_equal  # NOQA

from cupy.core.fusion import all  # NOQA
from cupy.core.fusion import any  # NOQA

# -----------------------------------------------------------------------------
# Mathematical functions
# -----------------------------------------------------------------------------
from cupy.core.fusion import arccos  # NOQA
from cupy.core.fusion import arcsin  # NOQA
from cupy.core.fusion import arctan  # NOQA
from cupy.core.fusion import arctan2  # NOQA
from cupy.core.fusion import cos  # NOQA
from cupy.core.fusion import deg2rad  # NOQA
from cupy.core.fusion import degrees  # NOQA
from cupy.core.fusion import hypot  # NOQA
from cupy.core.fusion import rad2deg  # NOQA
from cupy.core.fusion import radians  # NOQA
from cupy.core.fusion import sin  # NOQA
from cupy.core.fusion import tan  # NOQA

from cupy.core.fusion import arccosh  # NOQA
from cupy.core.fusion import arcsinh  # NOQA
from cupy.core.fusion import arctanh  # NOQA
from cupy.core.fusion import cosh  # NOQA
from cupy.core.fusion import sinh  # NOQA
from cupy.core.fusion import tanh  # NOQA

from cupy.core.fusion import ceil  # NOQA
from cupy.core.fusion import fix  # NOQA
from cupy.core.fusion import floor  # NOQA
from cupy.core.fusion import rint  # NOQA
from cupy.core.fusion import trunc  # NOQA

from cupy.core.fusion import prod  # NOQA
from cupy.core.fusion import sum  # NOQA
from cupy.math.sumprod import cumprod  # NOQA
from cupy.math.sumprod import cumsum  # NOQA
from cupy.math.window import blackman  # NOQA
from cupy.math.window import hamming  # NOQA
from cupy.math.window import hanning  # NOQA


from cupy.core.fusion import exp  # NOQA
from cupy.core.fusion import exp2  # NOQA
from cupy.core.fusion import expm1  # NOQA
from cupy.core.fusion import log  # NOQA
from cupy.core.fusion import log10  # NOQA
from cupy.core.fusion import log1p  # NOQA
from cupy.core.fusion import log2  # NOQA
from cupy.core.fusion import logaddexp  # NOQA
from cupy.core.fusion import logaddexp2  # NOQA

from cupy.core.fusion import copysign  # NOQA
from cupy.core.fusion import frexp  # NOQA
from cupy.core.fusion import ldexp  # NOQA
from cupy.core.fusion import nextafter  # NOQA
from cupy.core.fusion import signbit  # NOQA

from cupy.core.fusion import add  # NOQA
from cupy.core.fusion import divide  # NOQA
from cupy.core.fusion import floor_divide  # NOQA
from cupy.core.fusion import fmod  # NOQA
from cupy.core.fusion import modf  # NOQA
from cupy.core.fusion import multiply  # NOQA
from cupy.core.fusion import negative  # NOQA
from cupy.core.fusion import power  # NOQA
from cupy.core.fusion import reciprocal  # NOQA
from cupy.core.fusion import remainder  # NOQA
from cupy.core.fusion import remainder as mod  # NOQA
from cupy.core.fusion import subtract  # NOQA
from cupy.core.fusion import true_divide  # NOQA

from cupy.core.fusion import angle  # NOQA
from cupy.core.fusion import conj  # NOQA
from cupy.core.fusion import imag  # NOQA
from cupy.core.fusion import real  # NOQA

from cupy.core.fusion import abs  # NOQA
from cupy.core.fusion import absolute  # NOQA
from cupy.core.fusion import clip  # NOQA
from cupy.core.fusion import fmax  # NOQA
from cupy.core.fusion import fmin  # NOQA
from cupy.core.fusion import maximum  # NOQA
from cupy.core.fusion import minimum  # NOQA
from cupy.core.fusion import sign  # NOQA
from cupy.core.fusion import sqrt  # NOQA
from cupy.core.fusion import square  # NOQA

# -----------------------------------------------------------------------------
# Padding
# -----------------------------------------------------------------------------
pad = padding.pad.pad


# -----------------------------------------------------------------------------
# Sorting, searching, and counting
# -----------------------------------------------------------------------------
from cupy.sorting.count import count_nonzero  # NOQA
from cupy.sorting.search import flatnonzero  # NOQA
from cupy.sorting.search import nonzero  # NOQA

from cupy.core.fusion import where  # NOQA
from cupy.sorting.search import argmax  # NOQA
from cupy.sorting.search import argmin  # NOQA

from cupy.sorting.sort import argpartition  # NOQA
from cupy.sorting.sort import argsort  # NOQA
from cupy.sorting.sort import lexsort  # NOQA
from cupy.sorting.sort import msort  # NOQA
from cupy.sorting.sort import partition  # NOQA
from cupy.sorting.sort import sort  # NOQA

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
from cupy.core.fusion import amax  # NOQA
from cupy.core.fusion import amax as max  # NOQA
from cupy.core.fusion import amin  # NOQA
from cupy.core.fusion import amin as min  # NOQA
from cupy.statistics.order import nanmax  # NOQA
from cupy.statistics.order import nanmin  # NOQA
from cupy.statistics.order import percentile  # NOQA

from cupy.statistics.meanvar import average  # NOQA
from cupy.statistics.meanvar import mean  # NOQA
from cupy.statistics.meanvar import std  # NOQA
from cupy.statistics.meanvar import var  # NOQA

from cupy.statistics.histogram import bincount  # NOQA
from cupy.statistics.histogram import histogram  # NOQA

# -----------------------------------------------------------------------------
# Undocumented functions
# -----------------------------------------------------------------------------
from cupy.core import size  # NOQA

# -----------------------------------------------------------------------------
# CuPy specific functions
# -----------------------------------------------------------------------------

from cupy.util import clear_memo  # NOQA
from cupy.util import memoize  # NOQA

from cupy.core import ElementwiseKernel  # NOQA
from cupy.core import ReductionKernel  # NOQA


# The following function is left for backward compatibility.
# New CuPy specific routines should reside in cupyx package.
from cupy.ext.scatter import scatter_add  # NOQA

import cupyx


def asnumpy(a, stream=None):
    """Returns an array on the host memory from an arbitrary source array.

    Args:
        a: Arbitrary object that can be converted to :class:`numpy.ndarray`.
        stream (cupy.cuda.Stream): CUDA stream object. If it is specified, then
            the device-to-host copy runs asynchronously. Otherwise, the copy is
            synchronous. Note that if ``a`` is not a :class:`cupy.ndarray`
            object, then this argument has no effect.

    Returns:
        numpy.ndarray: Converted array on the host memory.

    """
    if isinstance(a, ndarray):
        return a.get(stream=stream)
    else:
        return numpy.asarray(a)


_cupy = sys.modules[__name__]


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
        if isinstance(arg, (ndarray, sparse.spmatrix)):
            return _cupy
    return numpy


fuse = fusion.fuse

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
    sys.stdout.write(str(cupyx.get_runtime_info()))
    sys.stdout.flush()
