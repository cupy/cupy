from libc cimport stdlib
from cython.operator cimport dereference

import cupy

from cupy.core.core cimport ndarray

from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport int64_t
from libc.stdint cimport uint64_t

from cupy.core.core cimport ndarray
from cupy.cuda cimport device as device_mod
from cupy.cuda cimport memory
cimport cpython
from cpython cimport pycapsule


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
    void (*deleter)(DLManagedTensor*)


cdef void deleter(DLManagedTensor* tensor) with gil:
    stdlib.free(tensor.dl_tensor.shape)
    cpython.Py_DECREF(<object>tensor.manager_ctx)
    stdlib.free(tensor)


cpdef object toDlpack(ndarray array):
    cdef DLManagedTensor* dlm_tensor = <DLManagedTensor*>stdlib.malloc(sizeof(DLManagedTensor))

    cdef size_t ndim = array._shape.size()
    cdef DLTensor* dl_tensor = &dlm_tensor.dl_tensor
    dl_tensor.data = array.data.ptr
    dl_tensor.ndim = ndim

    cdef int64_t* shape_strides = <int64_t*>stdlib.malloc(ndim * sizeof(int64_t) * 2)
    for n in range(ndim):
        shape_strides[n] = array._shape[n]
    dl_tensor.shape = shape_strides
    for n in range(ndim):
        shape_strides[n + ndim] = array._strides[n]

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

    return pycapsule.PyCapsule_New(dlm_tensor, 'dltensor', NULL)
