cimport numpy as cnp

import numpy

from cupy._core cimport _dtype


cdef extern from 'numpy/ndarraytypes.h':
    cdef int PyArray_Pack(cnp.dtype dtype, void *ptr, object value) except -1


cdef dict _typenames_base = {
    numpy.dtype('float64'): ('double', None),
    numpy.dtype('float32'): ('float', None),
    numpy.dtype('float16'): ('float16', '#include "cupy/float16.cuh"'),
    numpy.dtype('complex128'): ('thrust::complex<double>', None),
    numpy.dtype('complex64'): ('thrust::complex<float>', None),
    numpy.dtype('int64'): ('long long', None),
    numpy.dtype('int32'): ('int', None),
    numpy.dtype('int16'): ('short', None),
    numpy.dtype('int8'): ('signed char', None),
    numpy.dtype('uint64'): ('unsigned long long', None),
    numpy.dtype('uint32'): ('unsigned int', None),
    numpy.dtype('uint16'): ('unsigned short', None),
    numpy.dtype('uint8'): ('unsigned char', None),
    numpy.dtype('bool'): ('bool', None),
}


cdef object _numpy_bool = numpy.dtype(numpy.bool_)
cdef object _numpy_int32 = numpy.dtype(numpy.int32)
cdef object _numpy_int64 = numpy.dtype(numpy.int64)
cdef object _numpy_uint64 = numpy.dtype(numpy.uint64)
cdef object _numpy_float64 = numpy.dtype(numpy.float64)
cdef object _numpy_complex128 = numpy.dtype(numpy.complex128)


cpdef tuple get_typename_and_preamble(dtype):
    if dtype is None:
        raise ValueError('dtype is None')
    if dtype not in _typenames:
        dtype = _dtype.get_dtype(dtype).type
    return _typenames[dtype]


cpdef str get_typename(dtype, type_headers=None):
    """Fetch the C type name. Note that some names may require
    additionally headers to be included in order to be available.

    If not None, `type_headers` must be a set and the dtype preamble
    (i.e. this should be required headers) will be inserted.
    """
    if dtype is None:
        raise ValueError('dtype is None')
    if dtype not in _typenames:
        dtype = _dtype.get_dtype(dtype).type
    name, preamble = _typenames[dtype]
    if type_headers is not None and preamble is not None:
        type_headers.add(preamble)
    return name


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
        _typenames[t] = (t, None)

    try:
        import ml_dtypes
    except ImportError:
        pass
    else:
        dt = numpy.dtype(ml_dtypes.bfloat16)
        _dtype_kind_size_dict[dt] = ("V", 2)
        _typenames[dt] = (
            "bfloat16", '#include "cupy/bfloat16.cuh"')
        _dtype_kind_size_dict[dt.type] = ("V", 2)
        _typenames[dt.type] = (
            "bfloat16", '#include "cupy/bfloat16.cuh"')

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
    This is used as arguments for kernel launches were and may
    be cast to the kernel dtype via `apply_dtype()` (currently
    this will store the value a second time when needed).
    """
    ndim = 0

    def __init__(self, value, dtype=None):
        self.value = value
        if dtype is not None:
            self.descr = numpy.dtype(dtype)
            self.weak_t = False
        else:
            self.descr, self.weak_t = numpy_dtype_from_pyscalar(value)

            if self.descr is not None:
                pass  # Python scalar was processed
            elif isinstance(value, cnp.generic):
                self.descr = value.dtype
            else:
                # Future dtypes may have scalars where this is not the case
                # but for now, it should be fine.
                raise TypeError(f'Unsupported type {type(value)}')

        self._store_c_value()

    @staticmethod
    cdef CScalar from_int32(int32_t value):
        cdef CScalar self = CScalar.__new__(CScalar)
        self.value = None
        self.descr = _numpy_int32
        self.ptr = <void *>(self._data)
        (<int32_t *>(self.ptr))[0] = value
        return self

    cdef _store_c_value(self):
        # If we ever support dtypes larger than this (e.g. strings)
        # we will have to introduce a conditional allocation here and
        # should memset memory to NULL (must if dtype NEEDS_INIT).
        assert self.descr.itemsize < sizeof(self._data)
        self.ptr = <void *>(self._data)  # make sure ptr points to _data.

        # NOTE(seberg): This uses assignment logic, which is very subtly
        # different from casting by rejecting nan -> int. This is *only*
        # relevant for `casting="unsafe"` passed to ufuncs with `dtype=`.
        # It also means we fail for out of bound integers (NEP 50 change).
        PyArray_Pack(self.descr, self.ptr, self.value)

    cpdef apply_dtype(self, dtype):
        cdef cnp.dtype descr = cnp.dtype(dtype)
        if descr.flags & (0x01 | 0x04):
            # Can't support this, so make sure we raise appropriate error.
            _dtype.check_supported_dtype(descr, True)
            raise RuntimeError(f"Unsupported dtype {dtype} (but not raised?)")
        if descr == self.descr:
            self.descr = descr  # update dtype, may not be identical.
            return
        if self.value is None:
            # Internal/theoretical but e.g. from_int32 has no value
            raise RuntimeError("Cannot modify dtype if value is None.")

        self.descr = descr  # modify dtype if allocation succeeded
        self._store_c_value()

    cpdef get_numpy_type(self):
        return <object>(self.descr.typeobj)  # typeobj is the C-level .type


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
    elif dtype.name == "bfloat16":
        # NOTE(seberg): It would be nice to find a more extensible path here.
        float_repr = _get_cuda_scalar_repr(obj, numpy.dtype(numpy.float32))
        return f"bfloat16({float_repr})"

    raise TypeError(f'Unsupported dtype: {dtype}')
