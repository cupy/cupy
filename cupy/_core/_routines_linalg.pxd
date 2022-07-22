from cupy._core._carray cimport shape_t
from cupy._core.core cimport _ndarray_base


cpdef compute_type_to_str(compute_type)

cpdef get_compute_type(dtype)

cpdef _ndarray_base dot(_ndarray_base a, _ndarray_base b, _ndarray_base out=*)

cpdef _ndarray_base tensordot_core(
    _ndarray_base a, _ndarray_base b, _ndarray_base out, Py_ssize_t n,
    Py_ssize_t m, Py_ssize_t k, const shape_t& ret_shape)

cpdef _ndarray_base matmul(
    _ndarray_base a, _ndarray_base b, _ndarray_base out=*)


cpdef enum:
    COMPUTE_TYPE_TBD = 0
    COMPUTE_TYPE_DEFAULT = 1   # default
    COMPUTE_TYPE_PEDANTIC = 2  # disable algorithmic optimizations
    COMPUTE_TYPE_FP16 = 3      # allow converting inputs to FP16
    COMPUTE_TYPE_FP32 = 4      # allow converting inputs to FP32
    COMPUTE_TYPE_FP64 = 5      # allow converting inputs to FP64
    COMPUTE_TYPE_BF16 = 6      # allow converting inputs to BF16
    COMPUTE_TYPE_TF32 = 7      # allow converting inputs to TF32
