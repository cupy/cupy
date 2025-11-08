cimport cpython
import cython
from libc.stdint cimport intptr_t
from libcpp.string cimport string
from cupy._core.core cimport _ndarray_base
from cupy._core._scalar cimport CScalar as _cupy_scalar
include 'acl_types.pxi' # TODO: should not include in pxd file??

cdef str ASCEND_OP_PREFIX

cdef aclDataType numpy_to_acl_dtype(dtype,
    bint is_half_allowed=*, bint is_double_supported=*) except*
cdef aclTensor* cupy_ndarray_to_acl_tensor(_ndarray_base cupy_array) except*
cdef aclScalar* cupy_scalar_to_acl_scalar(_cupy_scalar s) except*

ctypedef fused sequence:
    list


# TODO: is size_t is the best type to pass C void* stream Pointer??
cdef aclError launch_general_func(str opname, sequence ins, sequence outs,
    list args, dict kargs, intptr_t stream_ptr) except *
cdef aclError launch_reduction_op(str opname, sequence ins, sequence outs,
    object axes, bint keepdims, dict kargs, intptr_t stream_ptr) except *