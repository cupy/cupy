import cupy


cdef void deleter(DLManagedTensor* tensor):
    pass


cdef object toDLPack(ndarray array):
    cdef DLContext ctx
    ctx.device_type = DLDeviceType.kDLGPU
    ctx.device_id = array.device.id

    cdef DLDataType dtype
    if cupy.issubdtype(array.dtype, cupy.integer):
        dtype.code = DLDataTypeCode.kDLInt
    elif cupy.issubdtype(array.dtype, cupy.unsignedinteger):
        dtype.code = DLDataTypeCode.kDLUInt
    elif cupy.issubdtype(array.dtype, cupy.floating):
        dtype.code = DLDataTypeCode.kDLFloat
    dtype.bits = array.dtype.itemsize * 8
    dtype.lanes = 1

    cdef DLTensor dl_tensor
    dl_tensor.data = <void *>array.data.ptr
    dl_tensor.ctx = ctx
    dl_tensor.ndim = array.ndim
    dl_tensor.dtype = dtype
    dl_tensor.shape = <int64_t*>&array._shape[0]
    dl_tensor.strides = <int64_t*>&array._strides[0]
    dl_tensor.byte_offset = 0

    cdef DLManagedTensor dlm_tensor
    dlm_tensor.dl_tensor = dl_tensor
    dlm_tensor.manager_ctx = &dl_tensor
    dlm_tensor.deleter = deleter

    return pycapsule.PyCapsule_New(&dlm_tensor, 'dltensor', NULL)


cpdef ndarray fromDLPack(object tensor):
    cdef DLManagedTensor* dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(tensor, 'dltensor')

    # Give 0 to size argument of the constructor of Memory
    # to prevent allocating any memory
    cdef memory.Memory mem = memory.Memory(0)
    mem.device = device_mod.Device(dlm_tensor.dl_tensor.ctx.device_id)
    mem.ptr = <size_t>dlm_tensor.dl_tensor.data

    # Calculates the size in bytes
    cdef int size = dlm_tensor.dl_tensor.shape[0]
    cdef int i = 1
    for i in range(1, dlm_tensor.dl_tensor.ndim):
        size = size * dlm_tensor.dl_tensor.shape[i]
    mem.size = dlm_tensor.dl_tensor.dtype.bits / 8 * size

    cdef memory.MemoryPointer mem_ptr = memory.MemoryPointer(mem, 0)
    if dlm_tensor.dl_tensor.dtype.code == DLDataTypeCode.kDLUInt:
        if dlm_tensor.dl_tensor.dtype.bits == 8:
            dtype = cupy.uint8
        elif dlm_tensor.dl_tensor.dtype.bits == 16:
            dtype = cupy.uint16
        elif dlm_tensor.dl_tensor.dtype.bits == 32:
            dtype = cupy.uint32
        elif dlm_tensor.dl_tensor.dtype.bits == 64:
            dtype = cupy.uint64
        else:
            raise TypeError('uint{} is not supported.'.format(dlm_tensor.dl_tensor.dtype.bits))
    elif dlm_tensor.dl_tensor.dtype.code == DLDataTypeCode.kDLInt:
        if dlm_tensor.dl_tensor.dtype.bits == 8:
            dtype = cupy.int8
        elif dlm_tensor.dl_tensor.dtype.bits == 16:
            dtype = cupy.int16
        elif dlm_tensor.dl_tensor.dtype.bits == 32:
            dtype = cupy.int32
        elif dlm_tensor.dl_tensor.dtype.bits == 64:
            dtype = cupy.int64
        else:
            raise TypeError('int{} is not supported.'.format(dlm_tensor.dl_tensor.dtype.bits))
    elif dlm_tensor.dl_tensor.dtype.code == DLDataTypeCode.kDLFloat:
        if dlm_tensor.dl_tensor.dtype.bits == 16:
            dtype = cupy.float16
        elif dlm_tensor.dl_tensor.dtype.bits == 32:
            dtype = cupy.float32
        elif dlm_tensor.dl_tensor.dtype.bits == 64:
            dtype = cupy.float64
        else:
            raise TypeError('float{} is not supported.'.format(dlm_tensor.dl_tensor.dtype.bits))
    else:
        raise TypeError('Unsupported dtype.')

    cdef int ndim = dlm_tensor.dl_tensor.ndim
    cdef int64_t[:] shape = <int64_t[:ndim]>dlm_tensor.dl_tensor.shape
    cdef ndarray array = ndarray(tuple(shape), dtype, mem_ptr)
    
    return array
