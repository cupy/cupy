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
    elif dtype.type_num == cnp.NPY_VOID and (
            not cnp.PyDataType_HASSUBARRAY(dtype) and dtype.itemsize != 0):
        # Support structured dtypes and void (not subarray here specifically).
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
        # I.e. an error may be raised when launching a kernel with this.
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
    elif dtype_char is not dtype and dtype.name == "bfloat16":
        # TODO(seberg): Better way to map this, doesn't support chars
        # due to ml_dtypes using 'E' (for now 2025-01)
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


def _make_aligned_dtype(
        dtype, *, int alignment=-1, bint recurse=True):
    # Internal version that also return the final alignment for recursion.
    cdef Py_ssize_t final_alignment = 1
    cdef Py_ssize_t itemsize = 0
    cdef Py_ssize_t curr_offset = 0
    cdef Py_ssize_t min_offset = 0  # offset before additional padding
    cdef cnp.dtype descr = cnp.dtype(dtype, align=True)

    if alignment != -1 and (alignment <= 0 or alignment.bit_count() != 1):
        raise ValueError("Reasonable alignments must be >0 and a power of 2.")

    if descr.type_num != cnp.NPY_VOID or (<object>descr).fields is None:
        # Allow this when `alignment=` is passed, we try to add the meatadat
        dtype_info = descr  # unchanged for now
        final_alignment = descr.alignment
    else:
        names = []
        offsets = []
        subdtypes = []

        for name, (subdtype, offset, *_) in (<object>descr).fields.items():
            # The fields tupe can contain a 4th title, we ignore it.
            # (A title is an alternative field name...)
            if offset < min_offset:
                raise ValueError(
                    "make_aligned_dtype() only supports well behaved "
                    "in order fields as it ignores field offsets).")

            # Keep track of field offset to reject non-ordered inputs.
            min_offset = offset + subdtype.itemsize

            subalignment = None
            if subdtype.metadata:
                subalignment = subdtype.metadata.get("__cuda_alignment__")

            if subalignment is not None:
                # __cuda_alignment__ is defined and overrides everything else.
                pass
            elif subdtype.num != cnp.NPY_VOID or subdtype.fields is None:
                subalignment = _scalar.get_cuda_alignment(subdtype)
            elif not recurse:
                # Must assume the alignment of the subdtype makes sense.
                subalignment = subdtype.alignment
            else:
                subdtype, subalignment = _make_aligned_dtype(
                    subdtype, recurse=recurse)

            if curr_offset % subalignment != 0:
                curr_offset += subalignment - (curr_offset % subalignment)

            final_alignment = max(final_alignment, subalignment)

            names.append(name)
            subdtypes.append(subdtype)
            offsets.append(curr_offset)
            curr_offset += subdtype.itemsize

        dtype_info = dict(names=names, offsets=offsets, formats=subdtypes,
                          itemsize=itemsize)

    metadata = {}
    if alignment != -1:
        if alignment >= final_alignment:
            final_alignment = alignment
            metadata = {"metadata": {"__cuda_alignment__": alignment}}
        else:
            raise ValueError(
                f"make_aligned_dtype(): given alignment={alignment} "
                f"smaller than minimum alignment {final_alignment}"
            )

    itemsize = (
        (curr_offset + final_alignment - 1) // final_alignment
        * final_alignment)

    if descr.type_num != cnp.NPY_VOID:
        if descr.itemsize != itemsize:
            raise ValueError(
                "Alignment larger than itemsize for non-structured dtype.")
    else:
        if descr.itemsize > itemsize:
            raise ValueError(
                "Input descriptor had larger itemsize than inferred.")
        dtype_info["itemsize"] = itemsize

    # Create a new dtype enforcing the newly computed offsets.
    return cnp.dtype(dtype_info, align=True, **metadata), final_alignment


def make_aligned_dtype(
        dtype, *, int alignment=-1, bint recurse=True):
    """Create a new structured dtype from a NumPy dtype or dtype-like with
    sufficient algnment for GPU use.

    Args:
        dtype: Data type specifier compatible with NumPy.
        alignment: Desired alignment of the resulting dtype. The dtype
            will be padded to ensure this alignment for CuPy.

            .. note::
                When the requested alignment is smaller than the minimal
                inferred one an error will be raised.
                When it is larger, CuPy will attach this alignment as
                metadata to the structured dtype. This is used for structured
                dtypes in the kernel and when nesting.

                Note that metadata may be lost in many operations.

        recurse: Whether to recurse into nested structures, defaults to True.
            When ``False``, the nested struct is assumed to be already
            sufficientlyaligned.

    Returns:
        cupy.ndarray: A view of the array with reduced dimensions.


    Notes:
        By default this function recurses into nested structures as if
        `alignment=-1` is passed for these.  You can nest a dtype with
        larger alignment by creating it with ``make_aligned_dtype()``.

        NumPy promotion (e.g. in concatenate) may "canonicalize" the dtype
        and drop the struct layout and CuPy alignment metadata.
    """
    return _make_aligned_dtype(
        dtype, alignment=alignment, recurse=recurse)[0]
