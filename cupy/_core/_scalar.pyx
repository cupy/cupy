from cpython cimport mem
from libc.stdint cimport int8_t
from libc.stdint cimport int16_t
from libc.stdint cimport int32_t
from libc.stdint cimport int64_t
from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport uint32_t
from libc.stdint cimport uint64_t

import numpy

from cupy._core cimport _dtype
from cupy._core import _dtype as _dtype_module
from cupy._core cimport internal


cdef union Scalar:
    bint bool_
    int8_t int8_
    int16_t int16_
    int32_t int32_
    int64_t int64_
    uint8_t uint8_
    uint16_t uint16_
    uint32_t uint32_
    uint64_t uint64_
    float float32_
    double float64_


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


cdef object _numpy_bool_ = numpy.bool_
cdef object _numpy_int8 = numpy.int8
cdef object _numpy_int16 = numpy.int16
cdef object _numpy_int32 = numpy.int32
cdef object _numpy_int64 = numpy.int64
cdef object _numpy_uint8 = numpy.uint8
cdef object _numpy_uint16 = numpy.uint16
cdef object _numpy_uint32 = numpy.uint32
cdef object _numpy_uint64 = numpy.uint64
cdef object _numpy_float16 = numpy.float16
cdef object _numpy_float32 = numpy.float32
cdef object _numpy_float64 = numpy.float64
cdef object _numpy_complex64 = numpy.complex64
cdef object _numpy_complex128 = numpy.complex128
cdef object _numpy_float_ = numpy.float_
cdef object _numpy_complex_ = numpy.complex_


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
    for i in _dtype_module.all_type_chars:
        d = numpy.dtype(i)
        t = d.type
        _typenames[t] = _typenames_base[d]
        k = ord(d.kind)
        _dtype_kind_size_dict[t] = (k, d.itemsize)
    # CUDA types
    for t in ('cudaTextureObject_t',):
        _typenames[t] = t


_setup_type_dict()


cdef set _python_scalar_type_set = {int, float, bool, complex}
cdef set _numpy_scalar_type_set = set(_typenames.keys())
cdef set scalar_type_set = _python_scalar_type_set | _numpy_scalar_type_set


_int_iinfo = numpy.iinfo(int)
cdef _int_min = _int_iinfo.min
cdef _int_max = _int_iinfo.max
cdef _int_type = _int_iinfo.dtype.type
cdef bint _use_int32 = _int_type != _numpy_int64
del _int_iinfo


cpdef _python_scalar_to_numpy_scalar(x):
    # Note that isinstance(x, int) matches with bool.
    typ = type(x)
    if typ is bool:
        numpy_type = _numpy_bool_
    elif typ is float:
        numpy_type = _numpy_float_
    elif typ is complex:
        numpy_type = _numpy_complex_
    else:
        if 0x8000000000000000 <= x:
            numpy_type = _numpy_uint64
        elif _use_int32 and (x < _int_min or _int_max < x):
            numpy_type = _numpy_int64
        else:
            # Generally `_int_type` is `numpy.int64`.
            # On Windows, it is `numpy.int32`.
            numpy_type = _int_type
    return numpy_type(x)


cdef class CScalar(CPointer):

    ndim = 0

    def __cinit__(self):
        self.ptr = mem.PyMem_Malloc(
            max(sizeof(Scalar), sizeof(double complex)))
        self.kind = 0
        self.size = -1

    def __dealloc__(self):
        mem.PyMem_Free(self.ptr)
        self.ptr = <void*>0

    @staticmethod
    cdef CScalar from_int32(int32_t value):
        cdef CScalar s = CScalar.__new__(CScalar)
        (<int32_t *>s.ptr)[0] = value
        s.kind = b'i'
        s.size = 4
        return s

    @staticmethod
    cdef CScalar from_numpy_scalar_with_dtype(object x, object dtype):
        cdef CScalar ret = CScalar._from_numpy_scalar(x)
        ret.apply_dtype(dtype)
        return ret

    @staticmethod
    cdef CScalar _from_python_scalar(object x):
        cdef CScalar ret = CScalar.__new__(CScalar)
        cdef Scalar* s = <Scalar*>ret.ptr
        typ = type(x)
        if typ is bool:
            s.bool_ = x
            ret.kind = b'b'
            ret.size = 1
        elif typ is float:
            s.float64_ = x
            ret.kind = b'f'
            ret.size = 8
        elif typ is complex:
            (<double complex*>ret.ptr)[0] = x
            ret.kind = b'c'
            ret.size = 16
        else:
            if 0x8000000000000000 <= x:
                s.uint64_ = x
                ret.kind = b'u'
            else:
                s.int64_ = x
                ret.kind = b'i'
            ret.size = 8
        return ret

    @staticmethod
    cdef CScalar _from_numpy_scalar(object x):
        cdef CScalar ret = CScalar.__new__(CScalar)
        cdef Scalar* s = <Scalar*>ret.ptr
        ret.kind = ord(x.dtype.kind)
        if ret.kind == b'i':
            s.int64_ = x
            ret.size = 8
        elif ret.kind == b'u':
            s.uint64_ = x
            ret.size = 8
        elif ret.kind == b'f':
            s.float64_ = x
            ret.size = 8
        elif ret.kind == b'b':
            s.bool_ = x
            ret.size = 1
        elif ret.kind == b'c':
            (<double complex*>ret.ptr)[0] = x
            ret.size = 16
        else:
            assert False
        return ret

    cpdef apply_dtype(self, dtype):
        cdef Scalar* s = <Scalar*>self.ptr
        if self.kind == b'b':
            val = s.bool_
            assert self.size == 1
        elif self.kind == b'c':
            assert self.size == 16
            val = (<double complex*>self.ptr)[0]
        else:
            assert self.size == 8
            if self.kind == b'i':
                val = s.int64_
            elif self.kind == b'u':
                val = s.uint64_
            elif self.kind == b'f':
                val = s.float64_
            else:
                assert False
        cdef char kind
        cdef int size
        kind, size = <tuple>_dtype_kind_size_dict[dtype]
        cdef int64_t val_i
        cdef uint64_t val_u
        if kind == b'b':
            s.bool_ = val
            assert size == 1
        elif kind == b'i':
            if self.kind == b'u':
                # avoid overflow exception
                val_i = s.uint64_
            else:
                val_i = val
            if size == 1:
                s.int8_ = val_i
            elif size == 2:
                s.int16_ = val_i
            elif size == 4:
                s.int32_ = val_i
            elif size == 8:
                s.int64_ = val_i
            else:
                assert False
        elif kind == b'u':
            if self.kind == b'i':
                # avoid overflow exception
                val_u = s.int64_
            else:
                val_u = val
            if size == 1:
                s.uint8_ = val_u
            elif size == 2:
                s.uint16_ = val_u
            elif size == 4:
                s.uint32_ = val_u
            elif size == 8:
                s.uint64_ = val_u
            else:
                assert False
        elif kind == b'f':
            if size == 2:
                s.uint16_ = internal.to_float16(<float>val)
            elif size == 4:
                s.float32_ = val
            elif size == 8:
                s.float64_ = val
            else:
                assert False
        elif kind == b'c':
            if size == 8:
                (<float complex*>self.ptr)[0] = val
            elif size == 16:
                (<double complex*>self.ptr)[0] = val
            else:
                assert False
        else:
            assert False
        self.kind = kind
        self.size = size

    cpdef get_numpy_type(self):
        if self.kind == b'b':
            return _numpy_bool_
        elif self.kind == b'i':
            if self.size == 1:
                return _numpy_int8
            elif self.size == 2:
                return _numpy_int16
            elif self.size == 4:
                return _numpy_int32
            elif self.size == 8:
                return _numpy_int64
        elif self.kind == b'u':
            if self.size == 1:
                return _numpy_uint8
            elif self.size == 2:
                return _numpy_uint16
            elif self.size == 4:
                return _numpy_uint32
            elif self.size == 8:
                return _numpy_uint64
        elif self.kind == b'f':
            if self.size == 2:
                return _numpy_float16
            elif self.size == 4:
                return _numpy_float32
            elif self.size == 8:
                return _numpy_float64
        elif self.kind == b'c':
            if self.size == 8:
                return _numpy_complex64
            elif self.size == 16:
                return _numpy_complex128
        assert False


cdef CScalar scalar_to_c_scalar(object x):
    # Converts a Python or NumPy scalar to a CScalar.
    # Returns None if the argument is not a scalar.
    typ = type(x)
    if typ in _python_scalar_type_set:
        return CScalar._from_python_scalar(x)
    elif typ in _numpy_scalar_type_set:
        return CScalar._from_numpy_scalar(x)
    return None


cdef object scalar_to_numpy_scalar(object x):
    # Converts a Python or NumPy scalar to a NumPy scalar.
    # Returns None if the argument is not a scalar.
    typ = type(x)
    if typ in _python_scalar_type_set:
        return _python_scalar_to_numpy_scalar(x)
    elif typ in _numpy_scalar_type_set:
        return x
    return None
