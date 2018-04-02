# DLPACK_VERSION: 010

from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport int64_t
from libc.stdint cimport uint64_t

from cupy.core.core cimport ndarray


ctypedef enum DLDeviceType:
    kDLCPU = 1
    kDLGPU = 2
    kDLCPUPinned = 3
    kDLOpenCL = 4
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10


ctypedef struct DLContext 'DLContext':
    DLDeviceType device_type
    int device_id


ctypedef enum DLDataTypeCode:
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2


ctypedef struct DLDataType 'DLDataType':
    uint8_t code
    uint8_t bits
    uint16_t lanes


ctypedef struct DLTensor 'DLTensor':
    void* data
    DLContext ctx
    int ndim
    DLDataType dtype
    int64_t* shape
    int64_t* strides
    uint64_t byte_offset


ctypedef struct DLManagedTensor 'DLManagedTensor':
    DLTensor dl_tensor
    void* manager_ctx
    void (*deleter)(DLManagedTensor*)


cdef void deleter(DLManagedTensor* tensor)
cdef object toDLPack(ndarray array)
cdef ndarray fromDLPack(DLManagedTensor* tensor)
