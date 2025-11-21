from cpython cimport mem

cimport numpy as cnp

import numpy

from cupy._core cimport _dtype


cdef extern from 'numpy/ndarraytypes.h':
    # Not exported by NumPy's `.pxd` due to being NumPy 2+ only.
    cdef int PyArray_Pack(cnp.dtype dtype, void *ptr, object value) except -1


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


cdef object _numpy_bool = numpy.dtype(numpy.bool_)
cdef object _numpy_int32 = numpy.dtype(numpy.int32)
cdef object _numpy_int64 = numpy.dtype(numpy.int64)
cdef object _numpy_uint64 = numpy.dtype(numpy.uint64)
cdef object _numpy_float64 = numpy.dtype(numpy.float64)
cdef object _numpy_complex128 = numpy.dtype(numpy.complex128)


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


_setup_type_dict()


cdef set _python_scalar_type_set = {int, float, bool, complex}
cdef set _numpy_scalar_type_set = set(_typenames.keys())
cdef set scalar_type_set = _python_scalar_type_set | _numpy_scalar_type_set


# Since NumPy 2 always true unless on 32bit (before windows was outlier)
assert numpy.dtype(int) == numpy.int64

cpdef tuple numpy_dtype_from_pyscalar(x):
    # Note that isinstance(x, int) matches with bool.
    typ = type(x)
    if typ is bool:
        return _numpy_bool, False
    elif typ is float:
        return _numpy_float64, float
    elif typ is complex:
        return _numpy_complex128, complex
    elif typ is int:
        if 0x8000000000000000 <= x:
            return _numpy_uint64, int
        else:
            return _numpy_int64, int

    return None, False


cdef class CScalar(CPointer):
    """Wrapper around NumPy/Python scalars to simplify internal
    processing and make a pointer to the data cleanly available.
    """
    ndim = 0

    def __init__(self, value, dtype=None):
        self.value = value
        if dtype is not None:
            self.value = value
            self.dtype = numpy.dtype(dtype)
            self.weak_t = False
        else:
            self.dtype, self.weak_t = numpy_dtype_from_pyscalar(value)

            if self.dtype is not None:
                pass  # Python scalar was processed
            elif isinstance(value, cnp.generic):
                self.dtype = value.dtype
                _dtype.check_supported_dtype(self.dtype, True)
            else:
                # Future dtypes may have scalars where this is not the case
                # but for now, it should be fine.
                raise TypeError(f'Unsupported type {type(value)}')

        self._store_c_value()

    cdef _free(self):
        if self.ptr != <void *>&(self._data):
            mem.PyMem_Free(self.ptr)
        self.ptr = <void*>0

    cdef _ensure_allocated(self, Py_ssize_t size, Py_ssize_t old_size):
        cdef void *new_ptr
        if old_size < 0:
            old_size = sizeof(self._data)
            self.ptr = <void *>&(self._data)

        if self.dtype.itemsize > old_size:
            new_ptr = mem.PyMem_Malloc(self.dtype.itemsize)
            if new_ptr == NULL:
                raise MemoryError()
            self._free()  # free old allocation (if there was one)
            self.ptr = new_ptr

    def __dealloc__(self):
        self._free()

    @staticmethod
    cdef CScalar from_int32(int32_t value):
        cdef CScalar self = CScalar.__new__(CScalar)
        self.value = None
        self.dtype = _numpy_int32
        self.ptr = <void *>&self._data
        (<int32_t *>self.ptr)[0] = value
        return self

    cdef _store_c_value(self):
        self._ensure_allocated(self.dtype.itemsize, -1)
        # NOTE(seberg): This uses assignment logic, which is very subtly
        # different from casting by rejecting nan -> int.
        # However, it means that we fail well for the weak scalar conversion.
        PyArray_Pack(self.dtype, self.ptr, self.value)

    cpdef apply_dtype(self, dtype):
        cdef cnp.dtype npdtype = cnp.dtype(dtype)
        if self.value is None:
            # Internal/theoretical but e.g. from_int32 has no value
            raise RuntimeError("Cannot modify dtype if value is None.")
        _dtype.check_supported_dtype(npdtype, True)
        self._ensure_allocated(npdtype.itemsize, self.dtype.itemsize)

        self.dtype = npdtype  # modify dtype if allocation succeeded
        self._store_c_value()

    cpdef get_numpy_type(self):
        # Use Python level `.type` lookup (different in cython level)
        # (This should use `type(self.dtype)` eventually.)
        return (<object>self.dtype).type


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
