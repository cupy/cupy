from cupy._core.core cimport _ndarray_base


cpdef cuda_compute_scan(_ndarray_base a, _ndarray_base result, dtype, str op)
