from cpython cimport mem

from libc.stdint cimport int8_t
from libc.stdint cimport int16_t
from libc.stdint cimport int32_t
from libc.stdint cimport int64_t
from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport uint32_t
from libc.stdint cimport uint64_t

cimport numpy as cnp

import numpy

from cupy._core cimport _dtype
from cupy._core cimport internal


cdef extern from 'numpy/ndarraytypes.h':
    cdef int PyArray_Pack(cnp.dtype dtype, void *ptr, object value) except -1


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
    numpy.dtype('complex128'): 'thrust::complex<double>',
    numpy.dtype('complex64'): 'thrust::complex<float>',
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
cdef object _numpy_float_ = numpy.float64
cdef object _numpy_complex_ = numpy.complex128


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
        k = ord(d.kind)
        _dtype_kind_size_dict[t] = (k, d.itemsize)
    # CUDA types
    for t in ('cudaTextureObject_t',):
        _typenames[t] = t

    try:
        import ml_dtypes
    except ImportError:
        pass
    else:
        dt = numpy.dtype(ml_dtypes.bfloat16)
        _dtype_kind_size_dict[dt] = ("V", 2)
        _typenames[dt.type] = "__nv_bfloat16"
        _dtype_kind_size_dict[dt] = ("V", 2)
        _typenames[dt.type] = "__nv_bfloat16"

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

    def __init__(self, Py_ssize_t size):
        self._init(size)

    cdef _init(self, Py_ssize_t size):
        if size > sizeof(self.data):
            self.ptr = mem.PyMem_Malloc(size)
            if self.ptr == NULL:
                raise MemoryError()
        else:
            self.ptr = <void *>&(self.data)
        self.kind = 0
        self.size = -1

    def __dealloc__(self):
        if self.ptr != <void *>&(self.data):
            mem.PyMem_Free(self.ptr)
        self.ptr = <void*>0

    @staticmethod
    cdef CScalar from_int32(int32_t value):
        cdef CScalar s = CScalar.__new__(CScalar)
        s._init(4)
        (<int32_t *>s.ptr)[0] = value
        s.kind = b'i'
        s.size = 4
        s.dtype = cnp.dtype("int32")
        return s

    @staticmethod
    cdef CScalar from_numpy_scalar_with_dtype(object x, object dtype_obj):
        # NOTE(seberg): This uses assignment logic, which is very subtly
        # different from casting by rejecting nan -> int. This should be fine.
        cdef CScalar ret = CScalar.__new__(CScalar)
        ret._init(16)
        cdef cnp.dtype dtype = cnp.dtype(dtype_obj)
        ret.kind = dtype.kind
        ret.size = dtype.itemsize
        ret.dtype = dtype

        _dtype.check_supported_dtype(dtype, True)

        PyArray_Pack(dtype, ret.ptr, x)
        return ret

    @staticmethod
    cdef CScalar _from_python_scalar(object x):
        cdef CScalar ret = CScalar.__new__(CScalar)
        ret._init(16)
        cdef Scalar* s = <Scalar*>ret.ptr
        typ = type(x)
        if typ is bool:
            s.bool_ = x
            ret.kind = b'b'
            ret.size = 1
            ret.dtype = cnp.dtype(bool)
        elif typ is float:
            s.float64_ = x
            ret.kind = b'f'
            ret.size = 8
            ret.dtype = cnp.dtype(float)
        elif typ is complex:
            (<double complex*>ret.ptr)[0] = x
            ret.kind = b'c'
            ret.size = 16
            ret.dtype = cnp.dtype(complex)
        else:
            if 0x8000000000000000 <= x:
                s.uint64_ = x
                ret.kind = b'u'
                ret.dtype = cnp.dtype("uint64")
            else:
                s.int64_ = x
                ret.kind = b'i'
                ret.dtype = cnp.dtype("int64")
            ret.size = 8
        return ret

    @staticmethod
    cdef CScalar _from_numpy_scalar(object x):
        cdef CScalar ret = CScalar.__new__(CScalar)
        ret._init(16)
        cdef cnp.dtype dtype = x.dtype
        ret.kind = dtype.kind
        ret.size = dtype.itemsize
        ret.dtype = dtype

        _dtype.check_supported_dtype(dtype, True)

        PyArray_Pack(dtype, ret.ptr, x)
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
        self.dtype = cnp.dtype(dtype)

    cpdef get_numpy_type(self):
        # Use Python level `.type` lookup (different in cython level)
        # (This should use `type(self.dtype)` eventually.)
        return (<object>self.dtype).type


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


cpdef str _get_cuda_scalar_repr(obj, dtype):
    if dtype.kind == 'b':
        return str(bool(obj)).lower()
    elif dtype.kind == 'i':
        if dtype.itemsize < 8:
            return str(int(obj))
        else:
            return str(int(obj)) + 'll'
    elif dtype.kind == 'u':
        if dtype.itemsize < 8:
            return str(int(obj)) + 'u'
        else:
            return str(int(obj)) + 'ull'
    elif dtype.kind == 'f':
        if dtype.itemsize < 8:
            if numpy.isnan(obj):
                return 'CUDART_NAN_F'
            elif numpy.isinf(obj):
                if obj > 0:
                    return 'CUDART_INF_F'
                else:
                    return '-CUDART_INF_F'
            else:
                return str(float(obj)) + 'f'
        else:
            if numpy.isnan(obj):
                return 'CUDART_NAN'
            elif numpy.isinf(obj):
                if obj > 0:
                    return 'CUDART_INF'
                else:
                    return '-CUDART_INF'
            else:
                return str(float(obj))
    elif dtype.kind == 'c':
        if dtype.itemsize == 8:
            return f'thrust::complex<float>({obj.real}, {obj.imag})'
        elif dtype.itemsize == 16:
            return f'thrust::complex<double>({obj.real}, {obj.imag})'

    raise TypeError(f'Unsupported dtype: {dtype}')
