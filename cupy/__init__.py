import functools as _functools
import sys as _sys

import numpy as _numpy

from cupy import _environment, _version

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


# Do not make `cupy.cupyx` available because it is confusing.
import cupyx as _cupyx
from cupy import cuda


def is_available():
    return cuda.is_available()


__version__ = _version.__version__


# =============================================================================
# Constants (borrowed from NumPy)
# =============================================================================
from numpy import (
    e,
    euler_gamma,
    inf,
    nan,
    newaxis,  # == None
    pi,
)

# `cupy.sparse` is deprecated in v8
from cupy import fft, linalg, polynomial, random, sparse, testing

# import class and function
from cupy._core import ndarray, ufunc

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
# Not supported by CuPy:
# from numpy import flexible
# from numpy import character
# -----------------------------------------------------------------------------
# Booleans
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Integers
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Unsigned integers
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Floating-point numbers
# -----------------------------------------------------------------------------
# from numpy import longfloat  # XXX
# Not supported by CuPy:
# from numpy import float96
# from numpy import float128
# -----------------------------------------------------------------------------
# Complex floating-point numbers
# -----------------------------------------------------------------------------
from numpy import (
    bool_,
    byte,
    cdouble,
    complex64,
    complex128,
    complexfloating,
    csingle,
    double,
    float16,
    float32,
    float64,
    floating,
    generic,
    half,
    inexact,
    int8,
    int16,
    int32,
    int64,
    int_,
    intc,
    integer,
    intp,
    longlong,
    number,
    short,
    signedinteger,
    single,
    ubyte,
    uint,
    uint8,
    uint16,
    uint32,
    uint64,
    uintc,
    uintp,
    ulonglong,
    unsignedinteger,
    ushort,
)
from numpy import complex64 as singlecomplex
from numpy import complex128 as cfloat
from numpy import complex128 as complex_
from numpy import float64 as float_

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
from cupy._creation.basic import (
    empty,
    empty_like,
    eye,
    full,
    full_like,
    identity,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from cupy._creation.from_data import (
    array,
    asanyarray,
    asarray,
    ascontiguousarray,
    copy,
    frombuffer,
    fromfile,
    fromfunction,
    fromiter,
    fromstring,
    genfromtxt,
    loadtxt,
)
from cupy._creation.matrix import diag, diagflat, tri, tril, triu, vander
from cupy._creation.ranges import (
    arange,
    linspace,
    logspace,
    meshgrid,
    mgrid,
    ogrid,
)

# -----------------------------------------------------------------------------
# Functional routines
# -----------------------------------------------------------------------------
from cupy._functional.piecewise import piecewise
from cupy._functional.vectorize import vectorize
from cupy._manipulation.add_remove import (
    append,
    delete,
    resize,
    trim_zeros,
    unique,
)

# -----------------------------------------------------------------------------
# Array manipulation routines
# -----------------------------------------------------------------------------
from cupy._manipulation.basic import copyto
from cupy._manipulation.dims import (
    atleast_1d,
    atleast_2d,
    atleast_3d,
    broadcast,
    broadcast_arrays,
    broadcast_to,
    expand_dims,
    squeeze,
)
from cupy._manipulation.join import (
    column_stack,
    concatenate,
    dstack,
    hstack,
    stack,
    vstack,
)
from cupy._manipulation.join import vstack as row_stack
from cupy._manipulation.kind import (
    asarray_chkfinite,
    asfarray,
    asfortranarray,
    require,
)
from cupy._manipulation.rearrange import flip, fliplr, flipud, roll, rot90
from cupy._manipulation.shape import ravel, reshape, shape
from cupy._manipulation.split import array_split, dsplit, hsplit, split, vsplit
from cupy._manipulation.tiling import repeat, tile
from cupy._manipulation.transpose import moveaxis, rollaxis, swapaxes, transpose
from cupy.lib._shape_base import apply_along_axis, put_along_axis

# Borrowed from NumPy
if hasattr(_numpy, 'broadcast_shapes'):  # NumPy 1.20
    from numpy import broadcast_shapes

# -----------------------------------------------------------------------------
# Binary operations
# -----------------------------------------------------------------------------
from cupy._binary.elementwise import (
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    invert,
    left_shift,
    right_shift,
)
from cupy._binary.packing import packbits, unpackbits


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


# Borrowed from NumPy
from numpy import (
    dtype,
    finfo,
    iinfo,
    index_exp,
    issubdtype,
    mintypecode,
    ndindex,
    promote_types,
    s_,
    typename,
)

from cupy._core.core import min_scalar_type

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
from cupy._indexing.generate import (
    c_,
    indices,
    ix_,
    mask_indices,
    r_,
    ravel_multi_index,
    tril_indices,
    tril_indices_from,
    triu_indices,
    triu_indices_from,
    unravel_index,
)
from cupy._indexing.indexing import (
    choose,
    compress,
    diagonal,
    extract,
    select,
    take,
    take_along_axis,
)
from cupy._indexing.insert import (
    diag_indices,
    diag_indices_from,
    fill_diagonal,
    place,
    put,
    putmask,
)
from cupy._indexing.iterate import flatiter
from cupy._io.formatting import (
    array2string,
    array_repr,
    array_str,
    format_float_positional,
    format_float_scientific,
)

# -----------------------------------------------------------------------------
# Input and output
# -----------------------------------------------------------------------------
from cupy._io.npz import load, save, savez, savez_compressed
from cupy._io.text import savetxt


def base_repr(number, base=2, padding=0):  # NOQA: F811 (needed to avoid redefinition of `number`)
    """Return a string representation of a number in the given base system.

    .. seealso:: :func:`numpy.base_repr`
    """
    return _numpy.base_repr(number, base, padding)


# Borrowed from NumPy
from numpy import get_printoptions, printoptions, set_printoptions

# -----------------------------------------------------------------------------
# Logic functions
# -----------------------------------------------------------------------------
from cupy._logic.comparison import allclose, array_equal, array_equiv, isclose
from cupy._logic.content import isfinite, isinf, isnan, isneginf, isposinf
from cupy._logic.truth import (
    in1d,
    intersect1d,
    isin,
    setdiff1d,
    setxor1d,
    union1d,
)
from cupy._logic.type_testing import (
    iscomplex,
    iscomplexobj,
    isfortran,
    isreal,
    isrealobj,
)

# -----------------------------------------------------------------------------
# Linear algebra
# -----------------------------------------------------------------------------
from cupy.linalg._einsum import einsum
from cupy.linalg._norms import trace
from cupy.linalg._product import (
    cross,
    dot,
    inner,
    kron,
    matmul,
    outer,
    tensordot,
    vdot,
)


def isscalar(element):
    """Returns True if the type of num is a scalar type.

    .. seealso:: :func:`numpy.isscalar`
    """
    return _numpy.isscalar(element)


# Borrowed from NumPy
from numpy import iterable

# -----------------------------------------------------------------------------
# Undocumented functions
# -----------------------------------------------------------------------------
from cupy._core import size
from cupy._logic.comparison import (
    equal,
    greater,
    greater_equal,
    less,
    less_equal,
    not_equal,
)
from cupy._logic.ops import logical_and, logical_not, logical_or, logical_xor
from cupy._logic.truth import all, alltrue, any, sometrue
from cupy._math.arithmetic import (
    add,
    angle,
    conjugate,
    divide,
    divmod,
    float_power,
    floor_divide,
    fmod,
    imag,
    modf,
    multiply,
    negative,
    positive,
    power,
    real,
    reciprocal,
    remainder,
    subtract,
    true_divide,
)
from cupy._math.arithmetic import conjugate as conj
from cupy._math.arithmetic import remainder as mod
from cupy._math.explog import (
    exp,
    exp2,
    expm1,
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logaddexp2,
)
from cupy._math.floating import copysign, frexp, ldexp, nextafter, signbit
from cupy._math.hyperbolic import arccosh, arcsinh, arctanh, cosh, sinh, tanh
from cupy._math.misc import (
    absolute,
    cbrt,
    clip,
    convolve,
    fabs,
    fmax,
    fmin,
    heaviside,
    interp,
    maximum,
    minimum,
    nan_to_num,
    real_if_close,
    sign,
    sqrt,
    square,
)
from cupy._math.misc import absolute as abs
from cupy._math.rational import gcd, lcm
from cupy._math.rounding import (
    around,
    ceil,
    fix,
    floor,
    rint,
    round,
    round_,
    trunc,
)
from cupy._math.special import i0, sinc
from cupy._math.sumprod import (
    cumprod,
    cumproduct,
    cumsum,
    diff,
    ediff1d,
    gradient,
    nancumprod,
    nancumsum,
    nanprod,
    nansum,
    prod,
    product,
    sum,
    trapz,
)

# -----------------------------------------------------------------------------
# Mathematical functions
# -----------------------------------------------------------------------------
from cupy._math.trigonometric import (
    arccos,
    arcsin,
    arctan,
    arctan2,
    cos,
    deg2rad,
    degrees,
    hypot,
    rad2deg,
    radians,
    sin,
    tan,
    unwrap,
)
from cupy._math.window import bartlett, blackman, hamming, hanning, kaiser

# -----------------------------------------------------------------------------
# Miscellaneous routines
# -----------------------------------------------------------------------------
from cupy._misc.byte_bounds import byte_bounds
from cupy._misc.memory_ranges import may_share_memory, shares_memory
from cupy._misc.who import who

# -----------------------------------------------------------------------------
# Padding
# -----------------------------------------------------------------------------
from cupy._padding.pad import pad

# -----------------------------------------------------------------------------
# Sorting, searching, and counting
# -----------------------------------------------------------------------------
from cupy._sorting.count import count_nonzero
from cupy._sorting.search import (
    argmax,
    argmin,
    argwhere,
    flatnonzero,
    nanargmax,
    nanargmin,
    nonzero,
    searchsorted,
    where,
)
from cupy._sorting.sort import (
    argpartition,
    argsort,
    lexsort,
    msort,
    partition,
    sort,
    sort_complex,
)

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------
from cupy._statistics.correlation import corrcoef, correlate, cov
from cupy._statistics.histogram import (
    bincount,
    digitize,
    histogram,
    histogram2d,
    histogramdd,
)
from cupy._statistics.meanvar import (
    average,
    mean,
    median,
    nanmean,
    nanmedian,
    nanstd,
    nanvar,
    std,
    var,
)
from cupy._statistics.order import (
    amax,
    amin,
    nanmax,
    nanmin,
    percentile,
    ptp,
    quantile,
)
from cupy._statistics.order import amax as max
from cupy._statistics.order import amin as min

# Borrowed from NumPy
# -----------------------------------------------------------------------------
# Classes without their own docs
# -----------------------------------------------------------------------------
from cupy.exceptions import (
    AxisError,
    ComplexWarning,
    ModuleDeprecationWarning,
    RankWarning,
    TooHardError,
    VisibleDeprecationWarning,
)

# ------------------------------------------------------------------------------
# Polynomial functions
# ------------------------------------------------------------------------------
from cupy.lib._polynomial import poly1d
from cupy.lib._routines_poly import (
    poly,
    polyadd,
    polyfit,
    polymul,
    polysub,
    polyval,
    roots,
)


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

# -----------------------------------------------------------------------------
# DLPack
# -----------------------------------------------------------------------------
from cupy._core import (
    ElementwiseKernel,
    RawKernel,
    RawModule,
    from_dlpack,
    fromDlpack,
)
from cupy._core._reduction import ReductionKernel
from cupy._util import clear_memo, memoize


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
    from numpy import DataSource, format_parser
else:
    from numpy.lib.npyio import DataSource
    from numpy.rec import format_parser  # type: ignore [no-redef]


# np 2.0: XXX shims for things removed without replacement
if _numpy.__version__ < "2":
    from numpy import (
        disp,
        find_common_type,
        get_array_wrap,
        safe_eval,
        set_string_function,
    )
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
