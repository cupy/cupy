# acl_tensor_converter.pyx
cimport cython
from libc.stdint cimport int64_t, uint64_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# 定义ACL数据类型和函数
cdef extern from "acl/acl.h" nogil:
    ctypedef enum aclDataType:
        ACL_DT_UNDEFINED = -1,
        ACL_FLOAT = 0,
        ACL_FLOAT16 = 1,
        ACL_INT8 = 2,
        ACL_INT32 = 3,
        ACL_UINT8 = 4,
        ACL_INT16 = 6,
        ACL_UINT16 = 7,
        ACL_UINT32 = 8,
        ACL_INT64 = 9,
        ACL_UINT64 = 10,
        ACL_DOUBLE = 11,
        ACL_BOOL = 12,
        ACL_COMPLEX32 = 33,
        ACL_COMPLEX64 = 16,
        ACL_COMPLEX128 = 17,
        ACL_BF16 = 27

    ctypedef enum aclFormat:
        ACL_FORMAT_ND = 1,
        ACL_FORMAT_NCHW = 2,
        ACL_FORMAT_NHWC = 3

cdef extern from "aclnn/opdev/common_types.h" nogil:
    ctypedef struct aclTensor:
        # 一个opaque的 aclTensor pointer类型， 可以传递pointer
        pass
