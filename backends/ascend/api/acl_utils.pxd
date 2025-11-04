cimport cpython
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

#cdef int register_acl_ufunc(str opname, object opcfunc) except*
# TODO: is size_t is the best type to pass C void* stream Pointer??
cdef aclError launch_general_func(str opname, list ins,
    list outs, list args, dict kargs, intptr_t stream_ptr) except *
cdef aclError launch_reduction_op(str opname, tuple ops, intptr_t stream_ptr) except *