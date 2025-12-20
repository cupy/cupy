cimport numpy as cnp

import numpy

from cupy._core cimport _dtype


cdef extern from 'numpy/ndarraytypes.h':
    """
    // PyArray_Pack is only defined if NPY_TARGET_VERSION=NPY_2_0_API_VERSION
    // which would hard disable 1.x support, so define it manually.
    #if NPY_FEATURE_VERSION < NPY_2_0_API_VERSION
    #define PyArray_Pack \
            (*(int (*)(PyArray_Descr *, void *, PyObject *)) \
        PyArray_API[65])
    #endif

    // Defined for NumPy 2.x builds, this define allows a local 1.x build.
    #if NPY_ABI_VERSION < 0x02000000
    static inline PyArray_ArrFuncs *
    PyDataType_GetArrFuncs(const PyArray_Descr *descr)
    {
        return descr->f;
    }
    #endif
    """
    # The above can be replaced with `NPY_TARGET_VERSION=NPY_2_0_API_VERSION`
    # as a define once NumPy 1.x is hard unsupported (raises on import).
    cdef int PyArray_Pack(cnp.dtype dtype, void *ptr, object value) except -1

    ctypedef int PyArray_SetItemFunc(object, void *ptr, void *arr) except -1
    cdef struct PyArray_ArrFuncs:
        PyArray_SetItemFunc setitem
    cdef PyArray_ArrFuncs *PyDataType_GetArrFuncs(cnp.dtype descr)


# Needed on C-side but Python side check should be safe enough.
cdef bint _IS_NUMPY_2 = numpy.__version__ >= '2'

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
        if _IS_NUMPY_2:
            PyArray_Pack(self.descr, self.ptr, self.value)
        elif self.descr.type in _dtype.all_type_chars_b:
            # Path can't support e.g. structured dtypes but we know it's OK
            # for all the above ones (last NULL needs an array sometimes).
            PyDataType_GetArrFuncs(self.descr).setitem(
                self.value, self.ptr, NULL)
        else:
            raise ValueError(f"Unsupported dtype {self.descr} (on NumPy 1.x)")

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

    raise TypeError(f'Unsupported dtype: {dtype}')
