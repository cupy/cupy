from cupy._core.core cimport _ndarray_base


cpdef enum cupy_cub_op:
    CUPY_CUB_SUM = 0
    CUPY_CUB_MIN = 1
    CUPY_CUB_MAX = 2
    CUPY_CUB_ARGMIN = 3
    CUPY_CUB_ARGMAX = 4
    CUPY_CUB_CUMSUM = 5
    CUPY_CUB_CUMPROD = 6
    CUPY_CUB_PROD = 7


# TODO(leofang): cimport these in other modules?
cpdef cub_reduction(_ndarray_base arr, op,
                    axis=*, dtype=*, _ndarray_base out=*, keepdims=*)
cpdef cub_scan(_ndarray_base arr, op)

cpdef bint _cub_device_segmented_reduce_axis_compatible(tuple, Py_ssize_t, str)
