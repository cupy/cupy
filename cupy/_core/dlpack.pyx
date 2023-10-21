cimport cpython  # NOQA

from libc cimport stdlib
from libc.stdint cimport uint8_t
from libc.stdint cimport uint16_t
from libc.stdint cimport int32_t
from libc.stdint cimport int64_t
from libc.stdint cimport uint64_t
from libc.stdint cimport intptr_t
from libcpp.vector cimport vector

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module
from cupy._core.core cimport _ndarray_base
from cupy.cuda cimport memory

import warnings

import cupy
import cupy._core.core as core


cdef extern from './include/cupy/_dlpack/dlpack.h' nogil:
    cdef int DLPACK_VERSION

    cdef enum DLDeviceType:
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

    cdef enum DLDataTypeCode:
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


def get_build_version():
    return str(DLPACK_VERSION)


cdef void pycapsule_deleter(object dltensor):
    cdef DLManagedTensor* dlm_tensor
    # Do not invoke the deleter on a used capsule
    if cpython.PyCapsule_IsValid(dltensor, 'dltensor'):
        dlm_tensor = <DLManagedTensor*>cpython.PyCapsule_GetPointer(
            dltensor, 'dltensor')
        dlm_tensor.deleter(dlm_tensor)


cdef void deleter(DLManagedTensor* tensor) with gil:
    if tensor.manager_ctx is NULL:
        return
    stdlib.free(tensor.dl_tensor.shape)
    cpython.Py_DECREF(<_ndarray_base>tensor.manager_ctx)
    tensor.manager_ctx = NULL
    stdlib.free(tensor)


# The name of this function is following the framework integration guide of
# TensorComprehensions.
cpdef object toDlpack(_ndarray_base array) except +:
    cdef DLManagedTensor* dlm_tensor = \
        <DLManagedTensor*>stdlib.malloc(sizeof(DLManagedTensor))

    cdef size_t ndim = array._shape.size()
    cdef DLTensor* dl_tensor = &dlm_tensor.dl_tensor
    cdef intptr_t data_ptr = array.data.ptr
    dl_tensor.data = <void*>data_ptr
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

    cdef DLDevice* device = &dl_tensor.device
    cdef bint is_managed
    cdef int dev_id = array.data.device_id
    if not runtime._is_hip_environment:
        attrs = runtime.pointerGetAttributes(data_ptr)
        is_managed = (attrs.type == runtime.memoryTypeManaged)
        if is_managed:
            device.device_type = kDLCUDAManaged
            dev_id = 0  # make it accessible on CPU too
        else:
            device.device_type = kDLCUDA
    else:
        device.device_type = kDLROCM
    device.device_id = dev_id

    cdef DLDataType* dtype = &dl_tensor.dtype
    if array.dtype.kind == 'u':
        dtype.code = <uint8_t>kDLUInt
    elif array.dtype.kind == 'i':
        dtype.code = <uint8_t>kDLInt
    elif array.dtype.kind == 'f':
        dtype.code = <uint8_t>kDLFloat
    elif array.dtype.kind == 'c':
        dtype.code = <uint8_t>kDLComplex
    elif array.dtype.kind == 'b':
        dtype.code = <uint8_t>kDLBool
    else:
        raise ValueError('Unknown dtype')
    dtype.lanes = <uint16_t>1
    dtype.bits = <uint8_t>(array.dtype.itemsize * 8)

    dlm_tensor.manager_ctx = <void*>array
    cpython.Py_INCREF(array)
    dlm_tensor.deleter = deleter

    return cpython.PyCapsule_New(dlm_tensor, 'dltensor', pycapsule_deleter)


# TODO(leofang): Support kDLCUDAPinned and kDLROCMPinned
cdef class DLPackMemory(memory.BaseMemory):

    """Memory object for a dlpack tensor.

    This does not allocate any memory.

    """

    cdef DLManagedTensor* dlm_tensor
    cdef object dltensor

    def __init__(self, object dltensor):
        cdef DLManagedTensor* dlm_tensor

        # sanity checks
        if not cpython.PyCapsule_IsValid(dltensor, 'dltensor'):
            raise ValueError('A DLPack tensor object cannot be consumed '
                             'multiple times')
        dlm_tensor = <DLManagedTensor*>cpython.PyCapsule_GetPointer(
            dltensor, 'dltensor')
        if runtime._is_hip_environment:
            if dlm_tensor.dl_tensor.device.device_type != kDLROCM:
                raise RuntimeError('CuPy is built against ROCm/HIP, different '
                                   'from the backend that backs the incoming '
                                   'DLPack tensor')
        else:
            if dlm_tensor.dl_tensor.device.device_type not in (
                    kDLCUDA, kDLCUDAManaged):
                raise RuntimeError('CuPy is built against CUDA, different '
                                   'from the backend that backs the incoming '
                                   'DLPack tensor')

        self.dltensor = dltensor
        self.dlm_tensor = dlm_tensor
        self.ptr = <intptr_t>dlm_tensor.dl_tensor.data
        if dlm_tensor.dl_tensor.device.device_type == kDLCUDAManaged:
            # look up the actual physical device as the id from
            # dl_tensor could be 0
            attrs = runtime.pointerGetAttributes(self.ptr)
            self.device_id = attrs.device
        else:
            self.device_id = dlm_tensor.dl_tensor.device.device_id

        cdef int n = 0, s = 0
        cdef int ndim = dlm_tensor.dl_tensor.ndim
        cdef int64_t* shape = dlm_tensor.dl_tensor.shape
        for s in shape[:ndim]:
            n += s
        self.size = dlm_tensor.dl_tensor.dtype.bits * n // 8

    def __dealloc__(self):
        cdef DLManagedTensor* dlm_tensor = self.dlm_tensor
        # dlm_tensor could be uninitialized if an error is raised in __init__
        if dlm_tensor != NULL:
            dlm_tensor.deleter(dlm_tensor)


# The name of this function is following the framework integration guide of
# TensorComprehensions.
cpdef _ndarray_base fromDlpack(object dltensor) except +:
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

    .. warning::

        This function is deprecated in favor of :func:`~cupy.from_dlpack` and
        will be removed in a future version of CuPy.

    .. warning::

        As of the DLPack v0.5 specification, it is implicitly assumed that
        the user is responsible to ensure the Producer and the Consumer are
        operating on the same stream.

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
    warnings.warn('This function is deprecated in favor of cupy.from_dlpack',
                  DeprecationWarning)
    return _dlpack_to_cupy_array(dltensor)


cdef inline _ndarray_base _dlpack_to_cupy_array(dltensor) except +:
    cdef DLPackMemory mem = DLPackMemory(dltensor)
    cdef DLDataType dtype = mem.dlm_tensor.dl_tensor.dtype
    cdef int bits = dtype.bits
    if dtype.lanes != 1:
        raise ValueError(f'vector dtypes (lanes={dtype.lanes}) is '
                         'not supported')
    if dtype.code == kDLUInt:
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
    elif dtype.code == kDLInt:
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
    elif dtype.code == kDLFloat:
        if bits == 16:
            cp_dtype = cupy.float16
        elif bits == 32:
            cp_dtype = cupy.float32
        elif bits == 64:
            cp_dtype = cupy.float64
        else:
            raise TypeError('float{} is not supported.'.format(bits))
    elif dtype.code == kDLComplex:
        # TODO(leofang): support complex32
        if bits == 64:
            cp_dtype = cupy.complex64
        elif bits == 128:
            cp_dtype = cupy.complex128
        else:
            raise TypeError('complex{} is not supported.'.format(bits))
    elif dtype.code == kDLBool:
        if bits == 8:
            cp_dtype = cupy.bool_
        else:
            raise TypeError(f'{bits}-bit bool is not supported')
    elif dtype.code == kDLBfloat:
        raise NotImplementedError('CuPy does not support bfloat16 yet')
    else:
        raise TypeError('Unsupported dtype. dtype code: {}'.format(dtype.code))

    mem_ptr = memory.MemoryPointer(mem, mem.dlm_tensor.dl_tensor.byte_offset)
    cdef int64_t ndim = mem.dlm_tensor.dl_tensor.ndim

    cdef int64_t* shape = mem.dlm_tensor.dl_tensor.shape
    cdef vector[Py_ssize_t] shape_vec
    shape_vec.assign(shape, shape + ndim)

    if mem.dlm_tensor.dl_tensor.strides is NULL:
        # Make sure this capsule will never be used again.
        cpython.PyCapsule_SetName(mem.dltensor, 'used_dltensor')
        return core.ndarray(shape_vec, cp_dtype, mem_ptr, strides=None)
    cdef int64_t* strides = mem.dlm_tensor.dl_tensor.strides
    cdef vector[Py_ssize_t] strides_vec
    for i in range(ndim):
        strides_vec.push_back(strides[i] * (bits // 8))

    # Make sure this capsule will never be used again.
    cpython.PyCapsule_SetName(mem.dltensor, 'used_dltensor')
    return core.ndarray(shape_vec, cp_dtype, mem_ptr, strides=strides_vec)


cpdef from_dlpack(array):
    """Zero-copy conversion between array objects compliant with the DLPack
    data exchange protocol.

    Args:
        array (object): an array object that implements two methods:
            ``__dlpack__()`` and ``__dlpack_device__()``.

    Returns:
        cupy.ndarray: a CuPy array that can be safely accessed on CuPy's
        current stream.

    .. note::
        This function is different from CuPy's legacy :func:`~cupy.fromDlpack`
        function. This function takes any object implementing the DLPack data
        exchange protocol, as well as a raw :class:`PyCapsule` object that
        contains the DLPack tensor as input (for backward compatibility),
        whereas :func:`~cupy.fromDlpack` only accepts :class:`PyCapsule`
        objects. If the input object is not compliant with the protocol, users
        are responsible to ensure data safety.

    .. seealso::
        :func:`numpy.from_dlpack`,
        `Python Specification for DLPack`_,
        `Data interchange mechanisms`_

    .. _Python Specification for DLPack:
        https://dmlc.github.io/dlpack/latest/python_spec.html
    .. _Data interchange mechanisms:
        https://data-apis.org/array-api/latest/design_topics/data_interchange.html
    """
    if not hasattr(array, '__dlpack_device__'):
        # backward compatibility: accept passing in a pycapsule
        dltensor = array
        return _dlpack_to_cupy_array(dltensor)
    else:
        dev_type, dev_id = array.__dlpack_device__()

    # CuPy is the consumer, so we provide our current stream to the producer
    if dev_type == <int>kDLCUDA or dev_type == <int>kDLCUDAManaged:
        prev_device = cupy.cuda.runtime.getDevice()
        try:
            cupy.cuda.runtime.setDevice(dev_id)
            assert not runtime._is_hip_environment
            stream = stream_module.get_current_stream_ptr()
            if stream == 0:
                stream = stream_module.get_default_stream_ptr()
            dltensor = array.__dlpack__(stream=stream)
        finally:
            cupy.cuda.runtime.setDevice(prev_device)
    elif dev_type == <int>kDLROCM:
        prev_device = cupy.cuda.runtime.getDevice()
        try:
            cupy.cuda.runtime.setDevice(dev_id)
            assert runtime._is_hip_environment
            stream = stream_module.get_current_stream_ptr()
            dltensor = array.__dlpack__(stream=stream)
        finally:
            cupy.cuda.runtime.setDevice(prev_device)
    elif dev_type == <int>kDLCPU:
        raise TypeError(
            'CPU arrays cannot be directly imported to CuPy. '
            'Use `cupy.array(numpy.from_dlpack(input))` instead.')
    else:
        # TODO(leofang): support kDLCUDAPinned etc
        dltensor = None
        raise TypeError(f'Unsupported array type: {dev_type}')

    return _dlpack_to_cupy_array(dltensor)
