from cupy._core.core cimport _ndarray_base

from libc.stdint cimport (
    uint8_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t
)


cdef extern from './include/cupy/_dlpack/dlpack.h' nogil:
    int DLPACK_MAJOR_VERSION
    int DLPACK_MINOR_VERSION
    int DLPACK_FLAG_BITMASK_READ_ONLY
    int DLPACK_FLAG_BITMASK_IS_COPIED

    ctypedef enum DLDeviceType:
        kDLCPU
        kDLCUDA
        kDLCUDAHost
        kDLOpenCL
        kDLVulkan
        kDLMetal
        kDLVPI
        kDLROCM
        kDLROCMHost
        kDLExtDev
        kDLCUDAManaged
        kDLOneAPI
        kDLWebGPU
        kDLHexagon

    ctypedef struct DLDevice:
        DLDeviceType device_type
        int32_t device_id

    ctypedef enum DLDataTypeCode:
        kDLInt
        kDLUInt
        kDLFloat
        kDLBfloat
        kDLComplex
        kDLBool

    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct DLTensor:
        void* data
        DLDevice device
        int32_t ndim
        DLDataType dtype
        int64_t* shape
        int64_t* strides
        uint64_t byte_offset

    ctypedef struct DLManagedTensor:
        DLTensor dl_tensor
        void* manager_ctx
        void (*deleter)(DLManagedTensor*)  # noqa: E211

    ctypedef struct DLPackVersion:
        uint32_t major
        uint32_t minor

    cdef struct DLManagedTensorVersioned:
        DLPackVersion version
        void* manager_ctx
        void (*deleter)(DLManagedTensorVersioned*)  # noqa: E211
        uint64_t flags
        DLTensor dl_tensor


cdef DLDevice get_dlpack_device(_ndarray_base array)
cpdef object toDlpack(
    _ndarray_base array, bint use_versioned=*, bint to_cpu=*,
    bint ensure_copy=*, stream=*) except +
cpdef _ndarray_base fromDlpack(object dltensor) except +
