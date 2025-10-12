cimport cpython
from libc.stdint cimport intptr_t
from libcpp.string cimport string
from cupy._core.core cimport _ndarray_base
from cupy._core._scalar cimport CScalar as _cupy_scalar
include 'acl_types.pxi'

cdef aclDataType numpy_to_acl_dtype(dtype,
    bint is_half_allowed=*, bint is_double_supported=*)
cdef aclTensor* cupy_ndarray_to_acl_tensor(_ndarray_base cupy_array) except*
cdef aclScalar* cupy_scalar_to_acl_scalar(_cupy_scalar s) except*

#cdef int register_acl_ufunc(str opname, object opcfunc) except*
# TODO: is size_t is the best type to pass C void* Pointer??
cdef aclError launch_acl_func(string opname, tuple ops, bint inplace, intptr_t stream) except *