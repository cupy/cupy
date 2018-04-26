from cython.operator cimport dereference

cimport cpython
from cpython cimport pycapsule

from libc cimport stdlib
from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport int64_t
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector

import cupy
from cupy.core.core cimport ndarray
from cupy.cuda cimport device as device_mod
from cupy.cuda cimport memory


cdef enum DLDeviceType:
    kDLCPU = 1
    kDLGPU = 2
    kDLCPUPinned = 3
    kDLOpenCL = 4
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10


cdef struct DLContext:
    DLDeviceType device_type
    int device_id


cdef enum DLDataTypeCode:
    kDLInt = <unsigned int>0
    kDLUInt = <unsigned int>1
    kDLFloat = <unsigned int>2


cdef struct DLDataType:
    uint8_t code
    uint8_t bits
    uint16_t lanes


cdef struct DLTensor:
    size_t data  # Safer than "void *"
    DLContext ctx
    int ndim
    DLDataType dtype
    int64_t* shape
    int64_t* strides
    uint64_t byte_offset


cdef struct DLManagedTensor:
    DLTensor dl_tensor
    void* manager_ctx
    void(*deleter)(DLManagedTensor*)


cdef void pycapsule_deleter(object dltensor):
    cdef DLManagedTensor* dlm_tensor
    try:
        dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(
            dltensor, 'used_dltensor')
    except:
        dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(
            dltensor, 'dltensor')
    deleter(dlm_tensor)


cdef void deleter(DLManagedTensor* tensor) with gil:
    if tensor.manager_ctx is NULL:
        return
    stdlib.free(tensor.dl_tensor.shape)
    cpython.Py_DECREF(<ndarray>tensor.manager_ctx)
    stdlib.free(tensor)
    tensor.manager_ctx = NULL


cpdef object toDlpack(ndarray array):
    cdef DLManagedTensor* dlm_tensor = \
        <DLManagedTensor*>stdlib.malloc(sizeof(DLManagedTensor))

    cdef size_t ndim = array._shape.size()
    cdef DLTensor* dl_tensor = &dlm_tensor.dl_tensor
    dl_tensor.data = array.data.ptr
    dl_tensor.ndim = ndim

    cdef int64_t* shape_strides = \
        <int64_t*>stdlib.malloc(ndim * sizeof(int64_t) * 2)
    for n in range(ndim):
        shape_strides[n] = array._shape[n]
    dl_tensor.shape = shape_strides
    for n in range(ndim):
        shape_strides[n + ndim] = array._strides[n] // array.dtype.itemsize

    dl_tensor.strides = shape_strides + ndim
    dl_tensor.byte_offset = 0

    cdef DLContext* ctx = &dl_tensor.ctx
    ctx.device_type = DLDeviceType.kDLGPU
    ctx.device_id = array.device.id

    cdef DLDataType* dtype = &dl_tensor.dtype
    if array.dtype.kind == 'u':
        dtype.code = <uint8_t>DLDataTypeCode.kDLUInt
    elif array.dtype.kind == 'i':
        dtype.code = <uint8_t>DLDataTypeCode.kDLInt
    elif array.dtype.kind == 'f':
        dtype.code = <uint8_t>DLDataTypeCode.kDLFloat
    else:
        raise ValueError('Unknown dtype')
    dtype.bits = <uint8_t>(array.dtype.itemsize * 8)
    dtype.lanes = <uint16_t>1

    dlm_tensor.manager_ctx = <void *>array
    cpython.Py_INCREF(array)
    dlm_tensor.deleter = deleter

    return pycapsule.PyCapsule_New(dlm_tensor, 'dltensor', pycapsule_deleter)


cdef class DLPackMemory(memory.Memory):

    """Memory object for a dlpack tensor.

    This does not allocate any memory.

    """

    cdef DLManagedTensor* dlm_tensor
    cdef object dltensor

    def __init__(self, object dltensor):
        self.dltensor = dltensor
        self.dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(
            dltensor, 'dltensor')
        self.device = cupy.cuda.Device(self.dlm_tensor.dl_tensor.ctx.device_id)
        self.ptr = self.dlm_tensor.dl_tensor.data
        cdef int n = 0
        cdef int ndim = self.dlm_tensor.dl_tensor.ndim
        cdef int64_t* shape = self.dlm_tensor.dl_tensor.shape
        for s in shape[:ndim]:
            n += s
        self.size = self.dlm_tensor.dl_tensor.dtype.bits * n // 8

        # Make sure this capsule will never be used again.
        pycapsule.PyCapsule_SetName(dltensor, 'used_dltensor')

    def __dealloc__(self):
        # DLPack tensor should be managed by the original creator
        self.ptr = 0


cpdef ndarray fromDlpack(object dltensor):
    mem = DLPackMemory(dltensor)

    cdef DLDataType dtype = mem.dlm_tensor.dl_tensor.dtype
    if dtype.code == DLDataTypeCode.kDLUInt:
        if dtype.bits == 8:
            cp_dtype = cupy.uint8
        elif dtype.bits == 16:
            cp_dtype = cupy.uint16
        elif dtype.bits == 32:
            cp_dtype = cupy.uint32
        elif dtype.bits == 64:
            cp_dtype = cupy.uint64
        else:
            raise TypeError('uint{} is not supported.'.format(dtype.bits))
    elif dtype.code == DLDataTypeCode.kDLInt:
        if dtype.bits == 8:
            cp_dtype = cupy.int8
        elif dtype.bits == 16:
            cp_dtype = cupy.int16
        elif dtype.bits == 32:
            cp_dtype = cupy.int32
        elif dtype.bits == 64:
            cp_dtype = cupy.int64
        else:
            raise TypeError('int{} is not supported.'.format(dtype.bits))
    elif dtype.code == DLDataTypeCode.kDLFloat:
        if dtype.bits == 16:
            cp_dtype = cupy.float16
        elif dtype.bits == 32:
            cp_dtype = cupy.float32
        elif dtype.bits == 64:
            cp_dtype = cupy.float64
        else:
            raise TypeError('float{} is not supported.'.format(dtype.bits))
    else:
        raise TypeError('Unsupported dtype. dtype code: {}'.format(dtype.code))

    mem_ptr = memory.MemoryPointer(mem, mem.dlm_tensor.dl_tensor.byte_offset)
    cdef int64_t ndim = mem.dlm_tensor.dl_tensor.ndim

    cdef int64_t* shape = mem.dlm_tensor.dl_tensor.shape
    cdef vector[Py_ssize_t] shape_vec
    shape_vec.assign(shape, shape + ndim)

    cdef int64_t* strides = mem.dlm_tensor.dl_tensor.strides
    cdef vector[Py_ssize_t] strides_vec
    for i in range(ndim):
        strides_vec.push_back(strides[i] * (dtype.bits // 8))

    cupy_array = ndarray(shape_vec, cp_dtype, mem_ptr)
    cupy_array._set_shape_and_strides(shape_vec, strides_vec)

    return cupy_array
