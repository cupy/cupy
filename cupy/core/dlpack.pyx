from libc.stdlib cimport malloc
from cython.operator cimport dereference

import cupy

from cupy.core.core cimport ndarray
from cupy.cuda.runtime cimport free


cdef void deleter(DLManagedTensor* tensor):
    free(<size_t>tensor.manager_ctx)


cpdef object toDlpack(ndarray array):
    cdef DLContext* ctx = <DLContext*>malloc(sizeof(DLContext))
    ctx.device_type = DLDeviceType.kDLGPU
    ctx.device_id = array.device.id

    cdef DLDataType* dtype = <DLDataType*>malloc(sizeof(DLDataType))
    if cupy.issubdtype(array.dtype, cupy.unsignedinteger):
        dtype.code = <uint8_t>DLDataTypeCode.kDLUInt
    elif cupy.issubdtype(array.dtype, cupy.integer):
        dtype.code = <uint8_t>DLDataTypeCode.kDLInt
    elif cupy.issubdtype(array.dtype, cupy.floating):
        dtype.code = <uint8_t>DLDataTypeCode.kDLFloat
    else:
        raise ValueError('Unknown dtype')
    dtype.bits = <uint8_t>(array.dtype.itemsize * 8)
    dtype.lanes = <uint16_t>1

    cdef DLTensor* dl_tensor = <DLTensor*>malloc(sizeof(DLTensor))
    cdef int ndim = array.ndim
    dl_tensor.data = <void *><size_t>array.data.ptr
    dl_tensor.ctx = dereference(ctx)
    dl_tensor.ndim = ndim
    dl_tensor.dtype = dereference(dtype)

    cdef int64_t* shape = <int64_t*>malloc(ndim * sizeof(int64_t))
    for n in range(ndim):
        shape[n] = array._shape[n]
    dl_tensor.shape = shape

    cdef int n_strides = len(array._strides)
    cdef int64_t* strides = <int64_t*>malloc(n_strides * sizeof(int64_t))
    for n in range(n_strides):
        strides[n] = array._strides[n]
    dl_tensor.strides = strides

    dl_tensor.byte_offset = 0

    cdef DLManagedTensor* dlm_tensor = <DLManagedTensor*>malloc(sizeof(DLManagedTensor))
    dlm_tensor.dl_tensor = dereference(dl_tensor)
    dlm_tensor.manager_ctx = array.get_pointer().ptr
    dlm_tensor.deleter = deleter

    return pycapsule.PyCapsule_New(dlm_tensor, 'dltensor', NULL)
