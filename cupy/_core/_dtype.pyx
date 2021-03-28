cimport cython  # NOQA
import numpy

from cupy_backends.cuda.api cimport runtime


all_type_chars = '?bhilqBHILQefdFD'
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
    if type(t) is _dtype:  # Exact type check
        return t
    ret = _dtype_dict.get(t, None)
    if ret is None:
        return _dtype(t)
    return ret[0]


@cython.profile(False)
cpdef tuple get_dtype_with_itemsize(t):
    if type(t) is _dtype:  # Exact type check
        return t, t.itemsize
    ret = _dtype_dict.get(t, None)
    if ret is None:
        t = _dtype(t)
        return t, t.itemsize
    return ret


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
    elif dtype_char == 'E' and is_half_allowed:
        # complex32, not supported in NumPy
        return runtime.CUDA_C_16F
    else:
        raise TypeError('dtype is not supported: {}'.format(dtype))
