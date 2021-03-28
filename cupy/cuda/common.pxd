# distutils: language = c++

# this mirrors the macro defined in cupy/_core/include/cupy/type_dispatcher.cuh
cdef enum:
    CUPY_TYPE_INT8 = 0
    CUPY_TYPE_UINT8 = 1
    CUPY_TYPE_INT16 = 2
    CUPY_TYPE_UINT16 = 3
    CUPY_TYPE_INT32 = 4
    CUPY_TYPE_UINT32 = 5
    CUPY_TYPE_INT64 = 6
    CUPY_TYPE_UINT64 = 7
    CUPY_TYPE_FLOAT16 = 8
    CUPY_TYPE_FLOAT32 = 9
    CUPY_TYPE_FLOAT64 = 10
    CUPY_TYPE_COMPLEX64 = 11
    CUPY_TYPE_COMPLEX128 = 12
    CUPY_TYPE_BOOL = 13


cpdef int _get_dtype_id(dtype) except -1
cpdef int _is_fp16_supported() except -2
