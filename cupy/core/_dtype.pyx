import numpy
import six


all_type_chars = '?bhilqBHILQefdFD'
# for c in '?bhilqBHILQefdFD':
#    print('#', c, '...', np.dtype(c).name)
# ? ... bool
# b ... int8
# h ... int16
# i ... int32
# l ... int64
# q ... int64
# B ... uint8
# H ... uint16
# I ... uint32
# L ... uint64
# Q ... uint64
# e ... float16
# f ... float32
# d ... float64
# F ... complex64
# D ... complex128

cdef dict _dtype_dict = {}


cdef _init_dtype_dict():
    for i in six.integer_types + (float, bool, complex, None):
        _dtype_dict[i] = numpy.dtype(i)
    for i in all_type_chars:
        dtype = numpy.dtype(i)
        _dtype_dict[i] = dtype
        _dtype_dict[dtype.type] = dtype


_init_dtype_dict()


cpdef get_dtype(t):
    if isinstance(t, numpy.dtype):
        return t
    ret = _dtype_dict.get(t, None)
    if ret is None:
        return numpy.dtype(t)
    return ret
