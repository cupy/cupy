cimport cpython  # NOQA

from libc cimport stdlib
from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport int64_t
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector

import cupy
from cupy.core.core cimport ndarray
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
        dlm_tensor = <DLManagedTensor *>cpython.PyCapsule_GetPointer(
            dltensor, 'used_dltensor')
        return             # we do not call a used capsule's deleter
    except Exception:
        dlm_tensor = <DLManagedTensor *>cpython.PyCapsule_GetPointer(
            dltensor, 'dltensor')
    deleter(dlm_tensor)


cdef void deleter(DLManagedTensor* tensor) with gil:
    if tensor.manager_ctx is NULL:
        return
    stdlib.free(tensor.dl_tensor.shape)
    cpython.Py_DECREF(<ndarray>tensor.manager_ctx)
    stdlib.free(tensor)
    tensor.manager_ctx = NULL


# The name of this function is following the framework integration guide of
# TensorComprehensions.
cpdef object toDlpack(ndarray array) except +:
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
    ctx.device_id = array.data.device_id

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

    return cpython.PyCapsule_New(dlm_tensor, 'dltensor', pycapsule_deleter)


cdef class DLPackMemory(memory.BaseMemory):

    """Memory object for a dlpack tensor.

    This does not allocate any memory.

    """

    cdef DLManagedTensor* dlm_tensor
    cdef object dltensor

    def __init__(self, object dltensor):
        self.dltensor = dltensor
        self.dlm_tensor = <DLManagedTensor *>cpython.PyCapsule_GetPointer(
            dltensor, 'dltensor')
        self.device_id = self.dlm_tensor.dl_tensor.ctx.device_id
        self.ptr = self.dlm_tensor.dl_tensor.data
        cdef int n = 0
        cdef int ndim = self.dlm_tensor.dl_tensor.ndim
        cdef int64_t* shape = self.dlm_tensor.dl_tensor.shape
        for s in shape[:ndim]:
            n += s
        self.size = self.dlm_tensor.dl_tensor.dtype.bits * n // 8

        # Make sure this capsule will never be used again.
        cpython.PyCapsule_SetName(dltensor, 'used_dltensor')

    def __dealloc__(self):
        self.dlm_tensor.deleter(self.dlm_tensor)


# The name of this function is following the framework integration guide of
# TensorComprehensions.
cpdef ndarray fromDlpack(object dltensor) except +:
    """Zero-copy conversion from a DLPack tensor to a :class:`~cupy.ndarray`.

    DLPack is a open in memory tensor structure proposed in this repository:
    `dmlc/dlpack <https://github.com/dmlc/dlpack>`_.

    This function takes a :class:`PyCapsule` object which contains a pointer to
    a DLPack tensor as input, and returns a :class:`~cupy.ndarray`. This
    function does not copy the data in the DLPack tensor but both
    DLPack tensor and :class:`~cupy.ndarray` have pointers which are pointing
    to the same memory region for the data.

    Args:
        dltensor (:class:`PyCapsule`): Input DLPack tensor which is
            encapsulated in a :class:`PyCapsule` object.

    Returns:
        array (:class:`~cupy.ndarray`): A CuPy ndarray.

    .. seealso::

        :meth:`cupy.ndarray.toDlpack` is a method for zero-copy conversion
        from a :class:`~cupy.ndarray` to a DLPack tensor (which is encapsulated
        in a :class:`PyCapsule` object).

    .. admonition:: Example

        >>> import cupy
        >>> array1 = cupy.array([0, 1, 2], dtype=cupy.float32)
        >>> dltensor = array1.toDlpack()
        >>> array2 = cupy.fromDlpack(dltensor)
        >>> cupy.testing.assert_array_equal(array1, array2)

    """
    mem = DLPackMemory(dltensor)

    cdef DLDataType dtype = mem.dlm_tensor.dl_tensor.dtype
    cdef int bits = dtype.bits
    if dtype.code == DLDataTypeCode.kDLUInt:
        if bits == 8:
            cp_dtype = cupy.uint8
        elif bits == 16:
            cp_dtype = cupy.uint16
        elif bits == 32:
            cp_dtype = cupy.uint32
        elif bits == 64:
            cp_dtype = cupy.uint64
        else:
            raise TypeError('uint{} is not supported.'.format(bits))
    elif dtype.code == DLDataTypeCode.kDLInt:
        if bits == 8:
            cp_dtype = cupy.int8
        elif bits == 16:
            cp_dtype = cupy.int16
        elif bits == 32:
            cp_dtype = cupy.int32
        elif bits == 64:
            cp_dtype = cupy.int64
        else:
            raise TypeError('int{} is not supported.'.format(bits))
    elif dtype.code == DLDataTypeCode.kDLFloat:
        if bits == 16:
            cp_dtype = cupy.float16
        elif bits == 32:
            cp_dtype = cupy.float32
        elif bits == 64:
            cp_dtype = cupy.float64
        else:
            raise TypeError('float{} is not supported.'.format(bits))
    else:
        raise TypeError('Unsupported dtype. dtype code: {}'.format(dtype.code))

    mem_ptr = memory.MemoryPointer(mem, mem.dlm_tensor.dl_tensor.byte_offset)
    cdef int64_t ndim = mem.dlm_tensor.dl_tensor.ndim

    cdef int64_t* shape = mem.dlm_tensor.dl_tensor.shape
    cdef vector[Py_ssize_t] shape_vec
    shape_vec.assign(shape, shape + ndim)

    if mem.dlm_tensor.dl_tensor.strides is NULL:
        return ndarray(shape_vec, cp_dtype, mem_ptr, strides=None)
    cdef int64_t* strides = mem.dlm_tensor.dl_tensor.strides
    cdef vector[Py_ssize_t] strides_vec
    for i in range(ndim):
        strides_vec.push_back(strides[i] * (bits // 8))

    return ndarray(shape_vec, cp_dtype, mem_ptr, strides=strides_vec)
