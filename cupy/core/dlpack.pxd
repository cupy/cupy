# DLPACK_VERSION: 010

from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport int64_t
from libc.stdint cimport uint64_t

from cupy.core.core cimport ndarray
from cupy.cuda cimport device as device_mod
from cupy.cuda cimport memory
from cpython cimport pycapsule


cdef extern from "dlpack/dlpack.h":

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
        kDLInt = <unsigned int>0
        kDLUInt = <unsigned int>1
        kDLFloat = <unsigned int>2


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
cpdef object toDlpack(ndarray array) except +
