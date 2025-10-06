cimport cpython

from cupy._core.core cimport _ndarray_base
from .acl_types cimport aclTensor, aclDataType 

cdef aclDataType numpy_to_acl_dtype(dtype,
    bint is_half_allowed=*, bint is_double_supported=*)
cdef aclTensor* cupy_ndarray_to_acl_tensor(_ndarray_base cupy_array) except*

cdef object register_acl_ufunc(str opname, object opcfunc) except*