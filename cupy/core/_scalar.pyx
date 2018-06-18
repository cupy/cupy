import numpy
import six

from cupy.core import _dtype
from cupy.core cimport internal


cdef dict _typenames_base = {
    numpy.dtype('float64'): 'double',
    numpy.dtype('float32'): 'float',
    numpy.dtype('float16'): 'float16',
    numpy.dtype('complex128'): 'complex<double>',
    numpy.dtype('complex64'): 'complex<float>',
    numpy.dtype('int64'): 'long long',
    numpy.dtype('int32'): 'int',
    numpy.dtype('int16'): 'short',
    numpy.dtype('int8'): 'signed char',
    numpy.dtype('uint64'): 'unsigned long long',
    numpy.dtype('uint32'): 'unsigned int',
    numpy.dtype('uint16'): 'unsigned short',
    numpy.dtype('uint8'): 'unsigned char',
    numpy.dtype('bool'): 'bool',
}


cpdef str get_typename(dtype):
    if dtype is None:
        raise ValueError('dtype is None')
    if dtype not in _typenames:
        dtype = _dtype.get_dtype(dtype).type
    return _typenames[dtype]


cdef dict _typenames = {}
cdef dict _dtype_kind_size_dict = {}


cdef _setup_type_dict():
    cdef char k
    for i in _dtype.all_type_chars:
        d = numpy.dtype(i)
        t = d.type
        _typenames[t] = _typenames_base[d]
        k = (<const char*>d.kind)[0]
        _dtype_kind_size_dict[t] = (k, d.itemsize)


_setup_type_dict()


cdef set _python_scalar_type_set = set(
    six.integer_types + (float, bool, complex))
cdef set _numpy_scalar_type_set = set(_typenames.keys())


_int_iinfo = numpy.iinfo(int)
cdef long long _int_min = _int_iinfo.min
cdef long long _int_max = _int_iinfo.max
cdef _int_type = _int_iinfo.dtype.type
del _int_iinfo


cpdef _python_scalar_to_numpy_scalar(x):
    # Note that isinstance(x, six_integer_types) matches with bool.
    typ = type(x)
    if typ is bool:
        numpy_type = numpy.bool_
    elif typ is float:
        numpy_type = numpy.float_
    elif typ is complex:
        numpy_type = numpy.complex_
    else:
        if 0x8000000000000000 <= x:
            numpy_type = numpy.uint64
        elif x < _int_min or _int_max < x:
            numpy_type = numpy.int64
        else:
            # Generally `_int_type` is `numpy.int64`.
            # On Windows, it is `numpy.int32`.
            numpy_type = _int_type
    return numpy_type(x)


cdef class CScalar(CPointer):
    ndim = 0

    def __init__(self):
        self.ptr = <void*>&(self.val)
        self.kind = ' '
        self.size = -1

    cpdef apply_dtype(self, dtype):
        if self.kind == 'b':
            val = self.val.bool_
            assert self.size == 1
        else:
            assert self.size == 8
            if self.kind == 'i':
                val = self.val.int64_
            elif self.kind == 'u':
                val = self.val.uint64_
            elif self.kind == 'f':
                val = self.val.float64_
            elif self.kind == 'c':
                val = self.val.complex128_
            else:
                assert False
        cdef char kind
        cdef int size
        kind, size = <tuple>_dtype_kind_size_dict[dtype]
        cdef int64_t val_i
        cdef uint64_t val_u
        if kind == 'b':
            self.val.bool_ = val
            assert size == 1
        elif kind == 'i':
            val_i = val
            if size == 1:
                self.val.int8_ = val_i
            elif size == 2:
                self.val.int16_ = val_i
            elif size == 4:
                self.val.int32_ = val_i
            elif size == 8:
                self.val.int64_ = val_i
            else:
                assert False
        elif kind == 'u':
            val_u = val
            if size == 1:
                self.val.uint8_ = val_u
            elif size == 2:
                self.val.uint16_ = val_u
            elif size == 4:
                self.val.uint32_ = val_u
            elif size == 8:
                self.val.uint64_ = val_u
            else:
                assert False
        elif kind == 'f':
            if size == 2:
                self.val.uint16_ = <uint16_t>internal.to_float16(<float>val)
            elif size == 4:
                self.val.float32_ = val
            elif size == 8:
                self.val.float64_ = val
            else:
                assert False
        elif kind == 'c':
            if size == 8:
                self.val.complex64_ = val
            elif size == 16:
                self.val.complex128_ = val
            else:
                assert False
        else:
            assert False


cpdef CScalar _python_scalar_to_c_scalar(x):
    cdef CScalar ret = CScalar()
    typ = type(x)
    if typ is bool:
        ret.val.bool_ = x
        ret.kind = 'b'
        ret.size = 1
    elif typ is float:
        ret.val.float64_ = x
        ret.kind = 'f'
        ret.size = 8
    elif typ is complex:
        ret.val.complex128_ = x
        ret.kind = 'c'
        ret.size = 16
    else:
        if 0x8000000000000000 <= x:
            ret.val.uint64_ = x
            ret.kind = 'u'
            ret.size = 8
        else:
            ret.val.int64_ = x
            ret.kind = 'i'
            ret.size = 8
    return ret


cpdef CScalar _numpy_scalar_to_c_scalar(x):
    cdef CScalar ret = CScalar()
    ret.kind = (<const char*>x.dtype.kind)[0]
    if ret.kind == 'i':
        ret.val.int64_ = x
        ret.size = 8
    elif ret.kind == 'u':
        ret.val.uint64_ = x
        ret.size = 8
    elif ret.kind == 'f':
        ret.val.float64_ = x
        ret.size = 8
    elif ret.kind == 'b':
        ret.val.bool_ = x
        ret.size = 1
    elif ret.kind == 'c':
        ret.val.complex128_ = x
        ret.size = 16
    else:
        assert False
    return ret


cpdef convert_scalar(x, bint use_c_scalar):
    typ = type(x)
    if typ in _python_scalar_type_set:
        if use_c_scalar:
            return _python_scalar_to_c_scalar(x)
        return _python_scalar_to_numpy_scalar(x)
    elif typ in _numpy_scalar_type_set:
        if use_c_scalar:
            return _numpy_scalar_to_c_scalar(x)
        return x
    return None
