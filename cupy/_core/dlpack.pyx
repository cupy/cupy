cimport cpython  # NOQA

from libc cimport stdlib
from libc.stdint cimport intptr_t
from libcpp.vector cimport vector

from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module
from cupy._core.core cimport _ndarray_base
from cupy.cuda cimport memory

import warnings

import cupy
import cupy._core.core as core
from cupy.cuda cimport stream as py_stream_module


cdef const char* CAPSULE_NAME = "dltensor"
cdef const char* CAPSULE_NAME_VER = "dltensor_versioned"
cdef const char* USED_CAPSULE_NAME = "used_dltensor"
cdef const char* USED_CAPSULE_NAME_VER = "used_dltensor_versioned"

# The higest major and minor DLPack version currently implemented and that
# will be usually exported.
cdef uint32_t IMPL_VER_MAJOR = 1
cdef uint32_t IMPL_VER_MINOR = 0


cdef void pycapsule_deleter(object dltensor) noexcept:
    cdef DLManagedTensor* dlm_tensor
    cdef DLManagedTensorVersioned *dlm_tensor_ver

    # Do not invoke the deleter on a used capsule
    if cpython.PyCapsule_IsValid(dltensor, CAPSULE_NAME):
        dlm_tensor = <DLManagedTensor*>(
            cpython.PyCapsule_GetPointer(dltensor, CAPSULE_NAME))
        dlm_tensor.deleter(dlm_tensor)
    elif cpython.PyCapsule_IsValid(dltensor, CAPSULE_NAME_VER):
        dlm_tensor_ver = <DLManagedTensorVersioned*>(
            cpython.PyCapsule_GetPointer(dltensor, CAPSULE_NAME_VER))
        dlm_tensor_ver.deleter(dlm_tensor_ver)
    else:
        # No cleanup necessary, capsule was "consumed" (renamed).
        pass


cdef void deleter(DLManagedTensor* tensor) noexcept with gil:
    # Delete fully initialized DLManagedTensor
    stdlib.free(tensor.dl_tensor.shape)
    cpython.Py_DECREF(<_ndarray_base>tensor.manager_ctx)
    tensor.manager_ctx = NULL
    stdlib.free(tensor)


cdef void deleter_ver(DLManagedTensorVersioned* tensor) noexcept with gil:
    # Delete fully initialized DLManagedTensorVersioned
    stdlib.free(tensor.dl_tensor.shape)
    cpython.Py_DECREF(<_ndarray_base>tensor.manager_ctx)
    tensor.manager_ctx = NULL
    stdlib.free(tensor)


cdef uint8_t get_dlpack_dtype_code(dtype) except? 255:
    """Convert NumPy/CuPy to dlpack dtype (kind, without bitsize).
    """
    cdef char kind = ord(dtype.kind)

    if kind == b'u':
        return <uint8_t>kDLUInt
    elif kind == b'i':
        return <uint8_t>kDLInt
    elif kind == b'f':
        return <uint8_t>kDLFloat
    elif kind == b'c':
        return <uint8_t>kDLComplex
    elif kind == b'b':
        return <uint8_t>kDLBool
    else:
        raise BufferError('dtype is not supported for dlpack export')


cdef DLDevice get_dlpack_device(_ndarray_base array):
    cdef DLDevice device
    cdef bint is_managed

    device.device_id = array.data.device_id

    if not runtime._is_hip_environment:
        attrs = runtime.pointerGetAttributes(array.data.ptr)
        is_managed = (attrs.type == runtime.memoryTypeManaged)
        if is_managed:
            device.device_type = kDLCUDAManaged
        else:
            device.device_type = kDLCUDA
    else:
        device.device_type = kDLROCM

    return device


# The name of this function is following the framework integration guide of
# TensorComprehensions.
cpdef object toDlpack(
    _ndarray_base array, bint use_versioned=False, bint to_cpu=False,
    bint ensure_copy=False, stream=None
):
    """Create a dlpack capsule for an array.

    Parameters
    ----------
    array : ndarray
        The array to export
    use_versioned : bool
        Whether to use the versioned struct.  In the future this may be
        which version to use.
    to_cpu : bool
        Whether we should make the data CPU available.
    ensure_copy : bool
        If `to_cpu` is True, whether a copy is requested/required.
    stream : None or stream
        Only used with `to_cpu`. The stream to use for making the data
        available to the CPU.
        If `None`, we make sure to synchronize to have the data available
        as soon as we return.  Otherwise, we use this stream to copy the
        data (as requested by the user).
    """
    cdef DLManagedTensor* dlm_tensor
    cdef DLManagedTensorVersioned* dlm_tensor_ver
    cdef DLTensor* dl_tensor
    cdef const char *capsule_name

    # Fetch dtype early (as this can raise a BufferError in theory)
    cdef uint8_t dtype_code = get_dlpack_dtype_code(array.dtype)
    cdef size_t dtype_itemsize = array.dtype.itemsize

    # Fetch device information since we need it to deal with CPU logic.
    cdef DLDevice device = get_dlpack_device(array)

    cdef int32_t ndim = array._shape.size()

    if not to_cpu:
        owner = array
    elif not ensure_copy and device.device_type == kDLCUDAManaged:
        # Managed memory is CPU accessible.  Note that the consumer may expect
        # `kDLCPU` here.  We only honor this request in spirit, but not
        # strictly (because e.g. NumPy will remember the managed part).
        owner = array
        if stream is None:
            # The user did not request a stream to synchronize on.  We have to
            # assume they don't even know this is GPU data, so must fully
            # synchronize on the current stream.
            py_stream_module.get_current_stream().synchronize()
    else:
        # We need to create a CPU copy.  Assumes owner.dtype == array.dtype.
        owner = array.get(
            stream=stream, order='A', out=None, blocking=stream is None)
        device.device_type = kDLCPU
        device.device_id = 0

    cdef void *dlm_tensor_ptr = stdlib.malloc(
        sizeof(DLManagedTensorVersioned) if use_versioned
        else sizeof(DLManagedTensor)
    )
    if dlm_tensor_ptr == NULL:
        raise MemoryError()

    # Note: could coalesce this with the previous allocation in principle
    cdef int64_t* shape_strides = <int64_t*>stdlib.malloc(
        ndim * sizeof(int64_t) * 2)
    if shape_strides == NULL:
        stdlib.free(dlm_tensor_ptr)
        raise MemoryError()

    # We need a different setup for versioned/unversioned when it comes to
    # the context/deleter and additional info in the newer versioned one.
    if use_versioned:
        dlm_tensor_ver = <DLManagedTensorVersioned*>dlm_tensor_ptr

        # dl_tensor is identically filled for versioned and unversioned:
        dl_tensor = &dlm_tensor_ver.dl_tensor
        capsule_name = CAPSULE_NAME_VER

        dlm_tensor_ver.manager_ctx = <void*>owner
        dlm_tensor_ver.deleter = deleter_ver

        # "Versioned" specific initialization:
        dlm_tensor_ver.version.major = IMPL_VER_MAJOR
        dlm_tensor_ver.version.minor = IMPL_VER_MINOR

        # CuPy arrays are writeable but may be copied if copying to the CPU.
        dlm_tensor_ver.flags = 0
        if owner is not array:
            dlm_tensor_ver.flags |= DLPACK_FLAG_BITMASK_IS_COPIED
    else:
        dlm_tensor = <DLManagedTensor*>dlm_tensor_ptr
        # dl_tensor is identically filled for versioned and unversioned:
        dl_tensor = &dlm_tensor.dl_tensor
        capsule_name = CAPSULE_NAME

        dlm_tensor.manager_ctx = <void*>owner
        dlm_tensor.deleter = deleter

    # Create capsule now. After the else, the capsule will clean up on error.
    # (Note that it is good to handle expected BufferErrors early.)
    try:
        capsule = cpython.PyCapsule_New(
            dlm_tensor_ptr, capsule_name, pycapsule_deleter)
    except BaseException:
        stdlib.free(dlm_tensor_ptr)
        stdlib.free(shape_strides)
        raise
    else:
        # Finalize dlm_tensor for `pycapsule_deleter`.  This else block must
        # never fail/raise and initialize everything used by the deleter.
        cpython.Py_INCREF(owner)
        dl_tensor.shape = shape_strides

    # And fill in all other fields (that are not part of the cleanup)
    dl_tensor.ndim = ndim
    dl_tensor.strides = shape_strides + ndim
    dl_tensor.byte_offset = 0

    dl_tensor.dtype.code = dtype_code
    dl_tensor.dtype.lanes = <uint16_t>1
    dl_tensor.dtype.bits = <uint8_t>(dtype_itemsize * 8)

    dl_tensor.device = device

    # Fill in the shape and strides information (depends on GPU vs CPU array).
    # (assumes strides are a multiple of itemsize, that should be OK for cupy.)
    if owner is array:
        dl_tensor.data = <void *><intptr_t>(array.data.ptr)

        for n in range(ndim):
            shape_strides[n] = array._shape[n]

        for n in range(ndim):
            shape_strides[n + ndim] = array._strides[n] // dtype_itemsize
    else:
        # Same as above, but we got a NumPy array, so go through Python
        # in the off-chance that the copy has changed the strides.
        dl_tensor.data = <void *><intptr_t>(owner.ctypes.data)
        shape = owner.shape
        strides = owner.strides

        for n in range(ndim):
            shape_strides[n] = shape[n]

        for n in range(ndim):
            shape_strides[n + ndim] = strides[n] // dtype_itemsize

    return capsule


# TODO(leofang): Support kDLCUDAPinned and kDLROCMPinned
cdef class DLPackMemory(memory.BaseMemory):

    """Memory object for a dlpack tensor.

    This does not allocate any memory.

    """

    cdef DLManagedTensor* dlm_tensor_unversioned
    cdef DLManagedTensorVersioned* dlm_tensor_ver
    cdef DLTensor* dl_tensor  # non owning reference

    def __init__(self, object dltensor):
        # First, take ownership of the contained DLManagedTensor(Versioned)
        # by copying it over and setting the capsule name to "used".
        if cpython.PyCapsule_IsValid(dltensor, CAPSULE_NAME):
            # Take ownership of the memory:
            self.dlm_tensor_unversioned = <DLManagedTensor*>(
                cpython.PyCapsule_GetPointer(dltensor, CAPSULE_NAME))
            cpython.PyCapsule_SetName(dltensor, USED_CAPSULE_NAME)

            self.dl_tensor = &self.dlm_tensor_unversioned.dl_tensor
        elif cpython.PyCapsule_IsValid(dltensor, CAPSULE_NAME_VER):
            # Take ownership of the memory:
            self.dlm_tensor_ver = <DLManagedTensorVersioned*>(
                cpython.PyCapsule_GetPointer(dltensor, CAPSULE_NAME_VER))
            cpython.PyCapsule_SetName(dltensor, USED_CAPSULE_NAME_VER)

            # When we have a versioned tensor, we need to verify the version
            # before further use.
            if self.dlm_tensor_ver.version.major > IMPL_VER_MAJOR:
                raise BufferError("DLPack exported too new major version.")

            # TODO(seberg): In principle we should raise an error if we got a
            #     readonly buffer.  But that might be disruptive :(.
            # if self.dlm_tensor_ver.flags & DLPACK_FLAG_BITMASK_READ_ONLY:
            #  raise BufferError("Buffer is readonly, but...")

            self.dl_tensor = &self.dlm_tensor_ver.dl_tensor
        else:
            # Be helpful in case a capsule is used twice somehow:
            raise ValueError(
                'A DLPack tensor object cannot be consumed multiple times '
                f'(or object was not a DLPack capsule). Got: {dltensor!r}')

        # Check if the device is compatible with the cupy runtime.
        if runtime._is_hip_environment:
            if self.dl_tensor.device.device_type != kDLROCM:
                raise RuntimeError('CuPy is built against ROCm/HIP, different '
                                   'from the backend that backs the incoming '
                                   'DLPack tensor')
        else:
            if self.dl_tensor.device.device_type not in (
                    kDLCUDA, kDLCUDAManaged):
                raise RuntimeError('CuPy is built against CUDA, different '
                                   'from the backend that backs the incoming '
                                   'DLPack tensor')

        self.ptr = <intptr_t>self.dl_tensor.data

        if self.dl_tensor.device.device_type == kDLCUDAManaged:
            # look up the actual physical device as the id from
            # dl_tensor could be 0
            attrs = runtime.pointerGetAttributes(self.ptr)
            self.device_id = attrs.device
        else:
            self.device_id = self.dl_tensor.device.device_id

        cdef int n = 0, s = 0
        cdef int ndim = self.dl_tensor.ndim
        cdef int64_t* shape = self.dl_tensor.shape
        for s in shape[:ndim]:
            n += s
        self.size = self.dl_tensor.dtype.bits * n // 8

    def __dealloc__(self):
        # dlm_tensor could be uninitialized if an error is raised in __init__
        if self.dlm_tensor_unversioned != NULL:
            if self.dlm_tensor_unversioned.deleter != NULL:
                self.dlm_tensor_unversioned.deleter(
                    self.dlm_tensor_unversioned)
        elif self.dlm_tensor_ver != NULL:
            if self.dlm_tensor_ver.deleter != NULL:
                self.dlm_tensor_ver.deleter(self.dlm_tensor_ver)


# The name of this function is following the framework integration guide of
# TensorComprehensions.
cpdef _ndarray_base fromDlpack(object dltensor):
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


cdef inline _ndarray_base _dlpack_to_cupy_array(dltensor):
    cdef DLPackMemory mem = DLPackMemory(dltensor)
    cdef DLDataType dtype = mem.dl_tensor.dtype
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

    mem_ptr = memory.MemoryPointer(mem, mem.dl_tensor.byte_offset)
    cdef int64_t ndim = mem.dl_tensor.ndim

    cdef int64_t* shape = mem.dl_tensor.shape
    cdef vector[Py_ssize_t] shape_vec
    shape_vec.assign(shape, shape + ndim)

    if mem.dl_tensor.strides is NULL:
        return core.ndarray(shape_vec, cp_dtype, mem_ptr, strides=None)
    cdef int64_t* strides = mem.dl_tensor.strides
    cdef vector[Py_ssize_t] strides_vec
    for i in range(ndim):
        strides_vec.push_back(strides[i] * (bits // 8))

    return core.ndarray(shape_vec, cp_dtype, mem_ptr, strides=strides_vec)


def from_dlpack(array, *, device=None, copy=None):
    """Zero-copy conversion between array objects compliant with the DLPack
    data exchange protocol.

    Args:
        array (object): an array object that implements two methods:
            ``__dlpack__()`` and ``__dlpack_device__()``.
        device (tuple): The dlpack device as a ``(device_type, device_id)``
            tuple.
        copy (boolean|None): Request export to never or always make a copy.
            By default (``None``) the exporter may or may not make a copy.

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

    if device is not None:
        # We can probably support the user picking a specific GPU in case that
        # is relevant to them for some reason.
        raise NotImplementedError("from_dlpack() does not support device yet.")

    # CuPy is the consumer, so we provide our current stream to the producer
    if dev_type == <int>kDLCUDA or dev_type == <int>kDLCUDAManaged:
        prev_device = cupy.cuda.runtime.getDevice()
        try:
            cupy.cuda.runtime.setDevice(dev_id)
            assert not runtime._is_hip_environment
            stream = stream_module.get_current_stream_ptr()
            if stream == 0:
                stream = stream_module.get_default_stream_ptr()

            # For backwards compatibility we catch TypeErrors for now.
            try:
                dltensor = array.__dlpack__(
                    stream=stream,
                    max_version=(IMPL_VER_MAJOR, IMPL_VER_MINOR),
                    copy=copy
                )
            except TypeError:
                if copy is not None:
                    raise
                dltensor = array.__dlpack__(stream=stream)
        finally:
            cupy.cuda.runtime.setDevice(prev_device)
    elif dev_type == <int>kDLROCM:
        prev_device = cupy.cuda.runtime.getDevice()
        try:
            cupy.cuda.runtime.setDevice(dev_id)
            assert runtime._is_hip_environment
            stream = stream_module.get_current_stream_ptr()

            # For backwards compatibility we catch TypeErrors for now.
            try:
                dltensor = array.__dlpack__(
                    stream=stream,
                    max_version=(IMPL_VER_MAJOR, IMPL_VER_MINOR),
                    copy=copy
                )
            except TypeError:
                if copy is not None:
                    raise
                dltensor = array.__dlpack__(stream=stream)
        finally:
            cupy.cuda.runtime.setDevice(prev_device)
    elif dev_type == <int>kDLCPU:
        raise TypeError(
            'CPU arrays cannot be directly imported to CuPy. '
            'Use `cupy.array(numpy.from_dlpack(input))` instead.')
    else:
        # TODO(leofang): support kDLCUDAPinned etc
        raise TypeError(f'Unsupported array type: {dev_type}')

    return _dlpack_to_cupy_array(dltensor)
