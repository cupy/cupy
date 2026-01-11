cimport cython  # NOQA

from . cimport _scalar

import numpy
import warnings

from cupy_backends.cuda.api cimport runtime
from cupy.exceptions import ComplexWarning


cdef str all_type_chars = '?bhilqBHILQefdFD'
cdef bytes all_type_chars_b = all_type_chars.encode()
# for c in '?bhilqBHILQefdFD':
#    print('#', c, '...', np.dtype(c).name)
# ? ... bool
# b ... int8
# h ... int16
# i ... int32
# l ... int64  (int32 in windows)
# q ... int64
# B ... uint8
# H ... uint16
# I ... uint32
# L ... uint64  (uint32 in windows)
# Q ... uint64
# e ... float16
# f ... float32
# d ... float64
# F ... complex64
# D ... complex128

cdef dict _dtype_dict = {}
cdef _dtype = numpy.dtype


cdef bint check_supported_dtype(cnp.dtype dtype, bint error) except -1:
    """ Returns true on success but otherwise raises an error. """

    if dtype.byteorder == b">":
        if not error:
            return False
        raise ValueError(
            f'Unsupported dtype {dtype} with big-endian byte-order')

    if dtype.type in all_type_chars_b:
        return True  # fast-path, these are always OK
    elif dtype.type == "V" and (<object>dtype).fields is not None:
        # Support structured dtypes (not subarray here specifically).
        # We don't really need to know anything about the dtype, but cannot
        # do references (copying back to CPU would be wrong).
        # Of course... the user may not be able to _do_ anything with it!
        if dtype.flags & (0x01 | 0x04):
            # Note, NumPy may (currently) flag this if a dtype has "holes"
            # such as `np.ones(10, dtype="i,O,i")[["f0", "f1"]]`.
            raise ValueError(
                f"Unsupported dtype {dtype} because it contains references "
                "which cannot be supported by CuPy.\n"
                "This may happen even if a dtype only contains basic types "
                "if the original array contained e.g. object dtype.")

        # We can represents the underlying bytes in CuPy, although it is very
        # possible that the included fields will not be usable in the end.
        # The simplest path is to inform users
        return True

    try:
        _scalar.get_typename(dtype)  # allow if we know a C typename.
        return True
    except (ValueError, KeyError):
        if not error:
            return False
        else:
            raise ValueError(f'Unsupported dtype {dtype}') from None


cdef _init_dtype_dict():
    for i in (int, float, bool, complex, None):
        dtype = _dtype(i)
        _dtype_dict[i] = (dtype, dtype.itemsize)
    for i in all_type_chars:
        dtype = _dtype(i)
        item = (dtype, dtype.itemsize)
        _dtype_dict[i] = item
        _dtype_dict[dtype.type] = item
    for i in {str(_dtype(i)) for i in all_type_chars}:
        dtype = _dtype(i)
        _dtype_dict[i] = (dtype, dtype.itemsize)


_init_dtype_dict()


@cython.profile(False)
cpdef get_dtype(t):
    ret = _dtype_dict.get(t, None)
    if ret is None:
        return _dtype(t)
    return ret[0]


@cython.profile(False)
cpdef tuple get_dtype_with_itemsize(t, bint check_support):
    # check_support for clarity, mainly array creation has to check.
    ret = _dtype_dict.get(t, None)
    if ret is not None:
        # Simple dtype request by user, always valid.
        return ret

    t = _dtype(t)
    if check_support:
        check_supported_dtype(t, error=True)
    return t, t.itemsize


cpdef int to_cuda_dtype(dtype, bint is_half_allowed=False) except -1:
    cdef str dtype_char
    try:
        dtype_char = dtype.char
    except AttributeError:
        dtype_char = dtype

    if dtype_char == 'e' and is_half_allowed:
        return runtime.CUDA_R_16F
    elif dtype_char == 'f':
        return runtime.CUDA_R_32F
    elif dtype_char == 'd':
        return runtime.CUDA_R_64F
    elif dtype_char == 'F':
        return runtime.CUDA_C_32F
    elif dtype_char == 'D':
        return runtime.CUDA_C_64F
    elif dtype_char == "E" and dtype.kind == "V":
        # TODO(seberg): Better way to map this?
        return runtime.CUDA_R_16BF
    elif dtype_char == 'E' and is_half_allowed:
        # complex32, not supported in NumPy
        return runtime.CUDA_C_16F
    else:
        raise TypeError('dtype is not supported: {}'.format(dtype))


cdef _numpy_can_cast = numpy.can_cast


cpdef void _raise_if_invalid_cast(
    from_dt, to_dt, str casting, argname="array data"
) except *:
    """Raise an error if a cast is not valid.  Also checks whether the cast
    goes from complex to real and warns if it does.

    The error raised can be customized by giving `obj`.  May pass a (lambda)
    function to avoid string construction on success.
    This function exists mainly to build a similar error everywhere.

    """
    if from_dt is to_dt:
        return

    to_dt = get_dtype(to_dt)  # may still be a type not a dtype instance

    if casting == "same_kind" and from_dt.kind == to_dt.kind:
        # same-kind is the most common casting used and for NumPy dtypes.
        return
    if _numpy_can_cast(from_dt, to_dt, casting):
        if casting == "unsafe" and from_dt.kind == "c" and to_dt.kind in "iuf":
            # Complex warning, we are dropping the imagine part:
            warnings.warn(
                'Casting complex values to real discards the imaginary part',
                ComplexWarning)

        return

    # Casting is not possible, raise the error
    if not isinstance(argname, str):
        argname = argname()
    raise TypeError(
        f'Cannot cast {argname} from {from_dt!r} to {to_dt!r} '
        f'according to the rule \'{casting}\'')


cdef dict dtype_format = {
    intern("?"): intern("?"),
    intern("b"): intern("b"),
    intern("h"): intern("h"),
    intern("i"): intern("i"),
    intern("l"): intern("l"),
    intern("q"): intern("q"),
    intern("B"): intern("B"),
    intern("H"): intern("H"),
    intern("I"): intern("I"),
    intern("L"): intern("L"),
    intern("Q"): intern("Q"),
    intern("e"): intern("e"),
    intern("f"): intern("f"),
    intern("d"): intern("d"),
    intern("F"): intern("Zf"),
    intern("D"): intern("Zd"),
}


@cython.profile(False)
cdef void populate_format(Py_buffer* buf, str dtype) except*:
    buf.format = dtype_format[dtype]
