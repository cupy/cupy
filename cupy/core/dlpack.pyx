import cupy


cdef void deleter(DLManagedTensor* tensor):
    del tensor.manager_ctx


cdef object toDLPack(ndarray array):
    cdef DLContext ctx
    ctx.device_type = DLDeviceType.kDLGPU
    ctx.device_id = array.device.id

    cdef DLDataType dtype
    if cupy.issubdtype(array.dtype, cupy.unsignedinteger):
        dtype.code = DLDataTypeCode.kDLUInt
    elif cupy.issubdtype(array.dtype, cupy.integer):
        dtype.code = DLDataTypeCode.kDLInt
    elif cupy.issubdtype(array.dtype, cupy.floating):
        dtype.code = DLDataTypeCode.kDLFloat
    dtype.bits = array.dtype.itemsize * 8
    dtype.lanes = 1

    cdef DLTensor dl_tensor
    dl_tensor.data = <void *><size_t>array.data.ptr
    dl_tensor.ctx = ctx
    dl_tensor.ndim = array.ndim
    dl_tensor.dtype = dtype
    dl_tensor.shape = <int64_t*>&array._shape[0]
    dl_tensor.strides = <int64_t*>&array._strides[0]
    dl_tensor.byte_offset = 0

    cdef DLManagedTensor dlm_tensor
    dlm_tensor.dl_tensor = dl_tensor
    dlm_tensor.manager_ctx = &array
    dlm_tensor.deleter = deleter

    return pycapsule.PyCapsule_New(&dlm_tensor, 'dltensor', NULL)
