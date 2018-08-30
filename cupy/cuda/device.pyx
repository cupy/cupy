# distutils: language = c++

import atexit

import six

from cupy.cuda cimport cublas
from cupy.cuda cimport cusparse
from cupy.cuda cimport runtime

try:
    from cupy.cuda import cusolver
    cusolver_enabled = True
except ImportError:
    cusolver_enabled = False


cpdef int get_device_id() except *:
    return runtime.getDevice()


cdef dict _cublas_handles = {}
cdef dict _cusolver_handles = {}
cdef dict _cusolver_sp_handles = {}
cdef dict _cusparse_handles = {}
cdef dict _compute_capabilities = {}


cpdef size_t get_cublas_handle() except *:
    dev_id = get_device_id()
    ret = _cublas_handles.get(dev_id, None)
    if ret is not None:
        return ret
    return Device().cublas_handle


cpdef size_t get_cusolver_handle() except *:
    dev_id = get_device_id()
    ret = _cusolver_handles.get(dev_id, None)
    if ret is not None:
        return ret
    return Device().cusolver_handle


cpdef get_cusolver_sp_handle():
    dev_id = get_device_id()
    if dev_id in _cusolver_sp_handles:
        return _cusolver_sp_handles[dev_id]
    return Device().cusolver_sp_handle


cpdef size_t get_cusparse_handle() except *:
    dev_id = get_device_id()
    ret = _cusparse_handles.get(dev_id, None)
    if ret is not None:
        return ret
    return Device().cusparse_handle


cpdef str get_compute_capability():
    dev_id = get_device_id()
    ret = _compute_capabilities.get(dev_id, None)
    if ret is not None:
        return ret
    return Device().compute_capability


cdef class Device:

    """Object that represents a CUDA device.

    This class provides some basic manipulations on CUDA devices.

    It supports the context protocol. For example, the following code is an
    example of temporarily switching the current device::

       with Device(0):
           do_something_on_device_0()

    After the *with* statement gets done, the current device is reset to the
    original one.

    Args:
        device (int or cupy.cuda.Device): Index of the device to manipulate. Be
            careful that the device ID (a.k.a. GPU ID) is zero origin. If it is
            a Device object, then its ID is used. The current device is
            selected by default.

    Attributes:
        id (int): ID of this device.

    """

    def __init__(self, device=None):
        if device is None:
            self.id = runtime.getDevice()
        else:
            self.id = int(device)

        self._device_stack = []

    def __int__(self):
        return self.id

    def __enter__(self):
        cdef int id = runtime.getDevice()
        self._device_stack.append(id)
        if self.id != id:
            self.use()
        return self

    def __exit__(self, *args):
        runtime.setDevice(self._device_stack.pop())

    def __repr__(self):
        return '<CUDA Device %d>' % self.id

    cpdef use(self):
        """Makes this device current.

        If you want to switch a device temporarily, use the *with* statement.

        """
        runtime.setDevice(self.id)

    cpdef synchronize(self):
        """Synchronizes the current thread to the device."""
        with self:
            runtime.deviceSynchronize()

    @property
    def compute_capability(self):
        """Compute capability of this device.

        The capability is represented by a string containing the major index
        and the minor index. For example, compute capability 3.5 is represented
        by the string '35'.

        """
        if self.id in _compute_capabilities:
            return _compute_capabilities[self.id]
        with self:
            major = runtime.deviceGetAttribute(75, self.id)
            minor = runtime.deviceGetAttribute(76, self.id)
            cc = '%d%d' % (major, minor)
            _compute_capabilities[self.id] = cc
            return cc

    @property
    def cublas_handle(self):
        """The cuBLAS handle for this device.

        The same handle is used for the same device even if the Device instance
        itself is different.

        """
        if self.id in _cublas_handles:
            return _cublas_handles[self.id]
        with self:
            handle = cublas.create()
            _cublas_handles[self.id] = handle
            return handle

    @property
    def cusolver_handle(self):
        """The cuSOLVER handle for this device.

        The same handle is used for the same device even if the Device instance
        itself is different.

        """
        if not cusolver_enabled:
            raise RuntimeError(
                'Current cupy only supports cusolver in CUDA 8.0')
        if self.id in _cusolver_handles:
            return _cusolver_handles[self.id]
        with self:
            handle = cusolver.create()
            _cusolver_handles[self.id] = handle
            return handle

    @property
    def cusolver_sp_handle(self):
        """The cuSOLVER Sphandle for this device.

        The same handle is used for the same device even if the Device instance
        itself is different.

        """
        if not cusolver_enabled:
            raise RuntimeError(
                'Current cupy only supports cusolver in CUDA 8.0')
        if self.id in _cusolver_sp_handles:
            return _cusolver_sp_handles[self.id]
        with self:
            handle = cusolver.spCreate()
            _cusolver_sp_handles[self.id] = handle
            return handle

    @property
    def cusparse_handle(self):
        """The cuSPARSE handle for this device.

        The same handle is used for the same device even if the Device instance
        itself is different.

        """
        if self.id in _cusparse_handles:
            return _cusparse_handles[self.id]
        with self:
            handle = cusparse.create()
            _cusparse_handles[self.id] = handle
            return handle

    @property
    def mem_info(self):
        """The device memory info.

        Returns:
            free: The amount of free memory, in bytes.
            total: The total amount of memory, in bytes.
        """
        with self:
            return runtime.memGetInfo()

    def __richcmp__(Device self, Device other, int op):
        if op == 0:
            return self.id < other.id
        if op == 1:
            return self.id <= other.id
        if op == 2:
            return self.id == other.id
        if op == 3:
            return self.id != other.id
        if op == 4:
            return self.id > other.id
        if op == 5:
            return self.id >= other.id
        return NotImplemented


def from_pointer(ptr):
    """Extracts a Device object from a device pointer.

    Args:
        ptr (ctypes.c_void_p): Pointer to the device memory.

    Returns:
        Device: The device whose memory the pointer refers to.

    """
    attrs = runtime.pointerGetAttributes(ptr)
    return Device(attrs.device)


@atexit.register
def destroy_cublas_handles():
    """Destroys the cuBLAS handles for all devices."""
    global _cublas_handles
    for handle in _cublas_handles.itervalues():
        cublas.destroy(handle)
    _cublas_handles = {}


@atexit.register
def destroy_cusparse_handles():
    """Destroys the cuSPARSE handles for all devices."""
    global _cusparse_handles
    for handle in six.itervalues(_cusparse_handles):
        cusparse.destroy(handle)
    _cusparse_handles = {}
