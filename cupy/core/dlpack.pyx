from cython.operator cimport dereference

cimport cpython
from cpython cimport pycapsule

from libc cimport stdlib
from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport int64_t
from libc.stdint cimport uint64_t

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
    void (*deleter)(DLManagedTensor*)


cdef void deleter(DLManagedTensor* tensor) with gil:
    stdlib.free(tensor.dl_tensor.shape)
    cpython.Py_DECREF(<object>tensor.manager_ctx)
    stdlib.free(tensor)


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


ctypedef void (*deleter_type)(DLManagedTensor*)


cdef class DLPackMemory(memory.Memory):

    """Memory object for a dlpack tensor.

    This does not allocate any memory.

    """

    cdef deleter_type deleter
    cdef DLManagedTensor* dlm_tensor

    def __init__(self, object dltensor):
        self.dlm_tensor = <DLManagedTensor *>pycapsule.PyCapsule_GetPointer(
            dltensor, 'dltensor')
        self.device = cupy.cuda.Device(self.dlm_tensor.dl_tensor.ctx.device_id)
        self.ptr = self.dlm_tensor.dl_tensor.data
        cdef int n = 0
        for s in self.dlm_tensor.dl_tensor.shape[
                :self.dlm_tensor.dl_tensor.ndim]:
            n += s
        self.size = self.dlm_tensor.dl_tensor.dtype.bits * n // 8
        self.deleter = self.dlm_tensor.deleter

    cpdef free(self):
        """Frees the dlpack tensor using its deleter.

        This function just calls ``deleter`` method of the DLManagedTensor.

        """
        self.deleter(self.dlm_tensor)

    def __dealloc__(self):
        # WHAT THE FUCK IS THIS?
        # if _exit_mode:
        #     return  # To avoid error at exit
        self.free()

cpdef ndarray fromDlpack(object dltensor):
    pass
