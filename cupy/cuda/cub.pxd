from cupy._core.core cimport ndarray


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
cpdef cub_reduction(ndarray arr, op,
                    axis=*, dtype=*, ndarray out=*, keepdims=*)
cpdef cub_scan(ndarray arr, op)

cpdef bint _cub_device_segmented_reduce_axis_compatible(tuple, Py_ssize_t, str)
