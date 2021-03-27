from cupy._core._carray cimport shape_t
from cupy._core.core cimport ndarray


cpdef compute_type_to_str(compute_type)

cpdef get_compute_type(dtype)

cpdef ndarray dot(ndarray a, ndarray b, ndarray out=*)

cpdef ndarray tensordot_core(
    ndarray a, ndarray b, ndarray out, Py_ssize_t n, Py_ssize_t m,
    Py_ssize_t k, const shape_t& ret_shape)

cpdef ndarray _matmul(ndarray a, ndarray b, ndarray out=*)


cpdef enum:
    COMPUTE_TYPE_TBD = 0
    COMPUTE_TYPE_DEFAULT = 1   # default
    COMPUTE_TYPE_PEDANTIC = 2  # disable algorithmic optimizations
    COMPUTE_TYPE_FP16 = 3      # allow converting inputs to FP16
    COMPUTE_TYPE_FP32 = 4      # allow converting inputs to FP32
    COMPUTE_TYPE_FP64 = 5      # allow converting inputs to FP64
    COMPUTE_TYPE_BF16 = 6      # allow converting inputs to BF16
    COMPUTE_TYPE_TF32 = 7      # allow converting inputs to TF32
