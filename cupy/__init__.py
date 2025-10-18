from __future__ import annotations

import functools as _functools
import sys as _sys

import numpy as _numpy

from cupy import _environment
from cupy import _version

# module alias to keep compatible for cupy.cuda code
import backends
cupy_backends = backends

_environment._detect_duplicate_installation()  # NOQA
_environment._setup_win32_dll_directory()  # NOQA
_environment._preload_library('cutensor')  # NOQA


try:
    from cupy import _core  # NOQA
except ImportError as exc:
    raise ImportError(f'''
================================================================
{_environment._diagnose_import_error()}

Original error:
  {type(exc).__name__}: {exc}
================================================================
''') from exc


from cupy import xpu  # NOQA
# Do not make `cupy.cupyx` available because it is confusing.
# TODO: ASCEND  usable only if this cupy module has been ported
#import cupyx as _cupyx  # NOQA


def is_available():
    return xpu.is_available()


__version__ = _version.__version__


# import class and function
from cupy._core import ndarray  # NOQA
from cupy._core import ufunc  # NOQA


# =============================================================================
# Constants (borrowed from NumPy)
# =============================================================================
from numpy import e  # NOQA
from numpy import euler_gamma  # NOQA
from numpy import inf  # NOQA
from numpy import nan  # NOQA
from numpy import newaxis  # == None  # NOQA
from numpy import pi  # NOQA

# APIs to be removed in NumPy 2.0.
# Remove these when bumping the baseline API to NumPy 2.0.
# https://github.com/cupy/cupy/pull/7800
PINF = Inf = Infinity = infty = inf  # NOQA
NINF = -inf  # NOQA
NAN = NaN = nan  # NOQA
PZERO = 0.0  # NOQA
NZERO = -0.0  # NOQA

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
from numpy import float64 as float_  # NOQA
# from numpy import longfloat  # NOQA   # XXX
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
from numpy import complex64 as singlecomplex  # NOQA
from numpy import cdouble  # NOQA
from numpy import complex128 as cfloat  # NOQA
from numpy import complex128 as complex_  # NOQA
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
# https://numpy.org/doc/stable/reference/routines.html
# =============================================================================


# -----------------------------------------------------------------------------
# Functional routines
# -----------------------------------------------------------------------------
# TODO:  ElementwiseKernel() not yet impl
#from cupy._functional.piecewise import piecewise  # NOQA
#from cupy._functional.vectorize import vectorize  # NOQA
from cupy.lib._shape_base import apply_along_axis  # NOQA
from cupy.lib._shape_base import apply_over_axes  # NOQA
from cupy.lib._shape_base import put_along_axis    # NOQA


# Borrowed from NumPy
if hasattr(_numpy, 'broadcast_shapes'):  # NumPy 1.20
    from numpy import broadcast_shapes  # NOQA


#from cupy._binary.packing import packbits  # NOQA
#from cupy._binary.packing import unpackbits  # NOQA


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


from cupy._core.core import min_scalar_type  # NOQA

from numpy import promote_types  # NOQA

from numpy import dtype  # NOQA

from numpy import finfo  # NOQA
from numpy import iinfo  # NOQA

from numpy import issubdtype  # NOQA

from numpy import mintypecode  # NOQA
from numpy import typename  # NOQA

# -----------------------------------------------------------------------------
# Optionally Scipy-accelerated routines
# -----------------------------------------------------------------------------
# TODO(beam2d): Implement it

# -----------------------------------------------------------------------------
# Discrete Fourier Transform
# -----------------------------------------------------------------------------
# TODO(beam2d): Implement it


# Borrowed from NumPy
from numpy import index_exp  # NOQA
from numpy import ndindex  # NOQA
from numpy import s_  # NOQA

# -----------------------------------------------------------------------------
# Input and output
# -----------------------------------------------------------------------------
from cupy._io.npz import load  # NOQA
from cupy._io.npz import save  # NOQA
from cupy._io.npz import savez  # NOQA
from cupy._io.npz import savez_compressed  # NOQA

from cupy._io.formatting import array_repr  # NOQA
from cupy._io.formatting import array_str  # NOQA
from cupy._io.formatting import array2string  # NOQA
from cupy._io.formatting import format_float_positional  # NOQA
from cupy._io.formatting import format_float_scientific  # NOQA

from cupy._io.text import savetxt  # NOQA


def base_repr(number, base=2, padding=0):  # NOQA (needed to avoid redefinition of `number`)
    """Return a string representation of a number in the given base system.

    .. seealso:: :func:`numpy.base_repr`
    """
    return _numpy.base_repr(number, base, padding)


# Borrowed from NumPy
from numpy import get_printoptions  # NOQA
from numpy import set_printoptions  # NOQA
from numpy import printoptions  # NOQA


def isscalar(element):
    """Returns True if the type of num is a scalar type.

    .. seealso:: :func:`numpy.isscalar`
    """
    return _numpy.isscalar(element)


# -----------------------------------------------------------------------------
# Miscellaneous routines
# -----------------------------------------------------------------------------
from cupy._misc.byte_bounds import byte_bounds  # NOQA
#from cupy._misc.memory_ranges import may_share_memory  # NOQA
#from cupy._misc.memory_ranges import shares_memory  # NOQA
#from cupy._misc.who import who  # NOQA

# Borrowed from NumPy
from numpy import iterable  # NOQA



# -----------------------------------------------------------------------------
# Exceptions and Warnings
# -----------------------------------------------------------------------------
from cupy import exceptions   # NOQA
from cupy.exceptions import AxisError  # NOQA
from cupy.exceptions import ComplexWarning  # NOQA
from cupy.exceptions import ModuleDeprecationWarning  # undocumented # NOQA
from cupy.exceptions import RankWarning  # NOQA
from cupy.exceptions import TooHardError  # NOQA
from cupy.exceptions import VisibleDeprecationWarning  # NOQA


# -----------------------------------------------------------------------------
# Undocumented functions
# -----------------------------------------------------------------------------
#from cupy._core import size  # NOQA


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

# -----------------------------------------------------------------------------
# DLPack
# -----------------------------------------------------------------------------

from cupy._core import fromDlpack  # NOQA
from cupy._core import from_dlpack  # NOQA


def asnumpy(a, stream=None, order='C', out=None, *, blocking=True):
    """Returns an array on the host memory from an arbitrary source array.

    Args:
        a: Arbitrary object that can be converted to :class:`numpy.ndarray`.
        stream (cupy.xpu.Stream): XPU stream object. If given, the
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
        return _core.array(a).get(
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

# TODO: ASCEND does not support fuse
#fuse = _core.fusion.fuse

disable_experimental_feature_warning = False


# set default allocator
_default_memory_pool = xpu.MemoryPool()
_default_pinned_memory_pool = xpu.PinnedMemoryPool()

xpu.set_allocator(_default_memory_pool.malloc)
xpu.set_pinned_memory_allocator(_default_pinned_memory_pool.malloc)


def get_default_memory_pool():
    """Returns CuPy default memory pool for GPU memory.

    Returns:
        cupy.xpu.MemoryPool: The memory pool object.

    .. note::
       If you want to disable memory pool, please use the following code.

       >>> cupy.xpu.set_allocator(None)

    """
    return _default_memory_pool


def get_default_pinned_memory_pool():
    """Returns CuPy default memory pool for pinned memory.

    Returns:
        cupy.xpu.PinnedMemoryPool: The memory pool object.

    .. note::
       If you want to disable memory pool, please use the following code.

       >>> cupy.xpu.set_pinned_memory_allocator(None)

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

# https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/numerictypes.py#L283-L322   # NOQA


def issubclass_(arg1, arg2):
    try:
        return issubclass(arg1, arg2)
    except TypeError:
        return False

# https://github.com/numpy/numpy/blob/v1.26.0/numpy/core/numerictypes.py#L229-L280   # NOQA


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


# https://github.com/numpy/numpy/blob/v1.26.0/numpy/core/numerictypes.py#L326C1-L355C1  # NOQA
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


# https://github.com/numpy/numpy/blob/v1.26.0/numpy/core/numerictypes.py#L457  # NOQA
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


# https://github.com/numpy/numpy/blob/v1.26.0/numpy/core/numerictypes.py#L184  # NOQA
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


# np 2.0: XXX shims for things newly added in np 2.0
if _numpy.__version__ < "2":
    from numpy import bool_ as bool  # NOQA
    from numpy import int_ as long  # NOQA
    from numpy import uint as ulong  # NOQA
else:
    from numpy import bool  # type: ignore [no-redef]  # NOQA
    from numpy import long  # type: ignore [no-redef]  # NOQA
    from numpy import ulong  # type: ignore [no-redef]  # NOQA


# np 2.0: XXX shims for things moved in np 2.0
if _numpy.__version__ < "2":
    from numpy import format_parser  # NOQA
    from numpy import DataSource     # NOQA
else:
    from numpy.rec import format_parser   # type: ignore [no-redef]  # NOQA
    from numpy.lib.npyio import DataSource  # NOQA


# np 2.0: XXX shims for things removed without replacement
if _numpy.__version__ < "2":
    from numpy import find_common_type   # NOQA
    from numpy import set_string_function  # NOQA
    from numpy import get_array_wrap  # NOQA
    from numpy import disp  # NOQA
    from numpy import safe_eval  # NOQA
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

# Lazy import testing. Set up after _embed_signatures so that we don't access
# any cupy.testing attributes. cupy.testing does not contain any ufuncs so it
# is not necessary to call _embed_signatures on it.
# https://docs.python.org/3/library/importlib.html#implementing-lazy-imports


def _lazy_import(name):
    import importlib.util
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    _sys.modules[name] = module
    loader.exec_module(module)
    return module


testing = _lazy_import("cupy.testing")
