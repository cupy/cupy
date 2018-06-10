import numpy
import six


cdef dict _dtype_dict


cdef _init_dtype_dict():
    global _dtype_dict
    _dtype_dict = {i: numpy.dtype(i)
                   for i in six.integer_types + (float, bool, complex, None)}
    for i in 'dfDFeqlihbQLIHB?':
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
