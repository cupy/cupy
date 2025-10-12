cimport cpython
from libcpp.string cimport string
from cupy._core.core cimport _ndarray_base
include 'acl_types.pxi'

cdef aclDataType numpy_to_acl_dtype(dtype,
    bint is_half_allowed=*, bint is_double_supported=*)
cdef aclTensor* cupy_ndarray_to_acl_tensor(_ndarray_base cupy_array) except*

#cdef int register_acl_ufunc(str opname, object opcfunc) except*
# TODO: is size_t is the best type to pass C void* Pointer??
cdef aclError launch_acl_func(string opname, tuple ops, bint inplace, intptr_t stream) except *