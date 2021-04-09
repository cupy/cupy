# distutils: language = c++

import threading

from cupy._core import syncdetect
from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda.api import runtime as runtime_module
from cupy_backends.cuda.libs import cublas
from cupy_backends.cuda.libs import cusolver
from cupy_backends.cuda.libs import cusparse
from cupy import _util


# This flag is kept for backward compatibility.
# It is always True as cuSOLVER library is always available in CUDA 8.0+.
cusolver_enabled = True

cdef object _thread_local = threading.local()

cdef dict _devices = {}
cdef dict _compute_capabilities = {}


cpdef int get_device_id() except? -1:
    return runtime.getDevice()


cpdef Device _get_device():
    dev_id = runtime.getDevice()
    ret = _devices.get(dev_id, None)
    if ret is None:
        ret = Device()
        _devices[dev_id] = ret
    return ret


cdef class Handle:
    def __init__(self, handle, destroy_func):
        self.handle = handle
        self._destroy_func = destroy_func

    def __dealloc__(self):
        self._destroy_func(self.handle)


cpdef intptr_t get_cublas_handle() except? 0:
    return _get_device().cublas_handle


cpdef intptr_t get_cusolver_handle() except? 0:
    return _get_device().cusolver_handle


cpdef intptr_t get_cusolver_sp_handle() except? 0:
    return _get_device().cusolver_sp_handle


cpdef intptr_t get_cusparse_handle() except? 0:
    return _get_device().cusparse_handle


cpdef str get_compute_capability():
    dev_id = get_device_id()
    ret = _compute_capabilities.get(dev_id, None)
    if ret is not None:
        return ret
    return Device().compute_capability


@_util.memoize()
def _get_attributes(device_id):
    """Return a dict containing all device attributes."""
    d = {}
    for k, v in runtime_module.__dict__.items():
        if k.startswith('cudaDevAttr'):
            try:
                name = k.replace('cudaDevAttr', '', 1)
                d[name] = runtime.deviceGetAttribute(v, device_id)
            except runtime_module.CUDARuntimeError as e:
                if e.status != runtime.errorInvalidValue:
                    raise
    return d


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

    @classmethod
    def from_pci_bus_id(cls, pci_bus_id):
        """Returns a new device instance based on a PCI Bus ID

        Args:
            pci_bus_id (str):
                The string for a device in the following format
                [domain]:[bus]:[device].[function] where domain, bus, device,
                and function are all hexadecimal values.
        Returns:
            device (Device):
                An instance of the Device class that has the PCI Bus ID as
                given by the argument pci_bus_id.
        """
        device_id = runtime.deviceGetByPCIBusId(pci_bus_id)
        return cls(device_id)

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
        syncdetect._declare_synchronize()
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
            major = runtime.deviceGetAttribute(
                runtime.deviceAttributeComputeCapabilityMajor, self.id)
            minor = runtime.deviceGetAttribute(
                runtime.deviceAttributeComputeCapabilityMinor, self.id)
            cc = '%d%d' % (major, minor)
            _compute_capabilities[self.id] = cc
            return cc

    def _get_handle(self, name, create_func, destroy_func):
        handles = getattr(_thread_local, name, None)
        if handles is None:
            handles = {}
            setattr(_thread_local, name, handles)
        handle = handles.get(self.id, None)
        if handle is not None:
            return handle.handle
        with self:
            handle = create_func()
            handles[self.id] = Handle(handle, destroy_func)
            return handle

    @property
    def cublas_handle(self):
        """The cuBLAS handle for this device.

        The same handle is used for the same device even if the Device instance
        itself is different.

        """
        return self._get_handle(
            'cublas_handles', cublas.create, cublas.destroy)

    @property
    def cusolver_handle(self):
        """The cuSOLVER handle for this device.

        The same handle is used for the same device even if the Device instance
        itself is different.

        """
        return self._get_handle(
            'cusolver_handles', cusolver.create, cusolver.destroy)

    @property
    def cusolver_sp_handle(self):
        """The cuSOLVER Sphandle for this device.

        The same handle is used for the same device even if the Device instance
        itself is different.

        """
        return self._get_handle(
            'cusolver_sp_handles', cusolver.spCreate, cusolver.spDestroy)

    @property
    def cusparse_handle(self):
        """The cuSPARSE handle for this device.

        The same handle is used for the same device even if the Device instance
        itself is different.

        """
        return self._get_handle(
            'cusparse_sp_handles', cusparse.create, cusparse.destroy)

    @property
    def mem_info(self):
        """The device memory info.

        Returns:
            free: The amount of free memory, in bytes.
            total: The total amount of memory, in bytes.
        """
        with self:
            return runtime.memGetInfo()

    @property
    def attributes(self):
        """A dictionary of device attributes.

        Returns:
            attributes (dict):
                Dictionary of attribute values with the names as keys.
                The string `cudaDevAttr` has been trimmed from the names.
                For example, the attribute corresponding to the enumerated
                value `cudaDevAttrMaxThreadsPerBlock` will have key
                `MaxThreadsPerBlock`.
        """
        return _get_attributes(self.id)

    @property
    def pci_bus_id(self):
        """A string of the PCI Bus ID

        Returns:
            pci_bus_id (str):
                Returned identifier string for the device in the following
                format [domain]:[bus]:[device].[function] where domain, bus,
                device, and function are all hexadecimal values.
        """
        return runtime.deviceGetPCIBusId(self.id)

    def __richcmp__(Device self, object other, int op):
        if op == 2:
            return isinstance(other, Device) and self.id == other.id
        if op == 3:
            return not (isinstance(other, Device) and self.id == other.id)
        if not isinstance(other, Device):
            return NotImplemented
        if op == 0:
            return self.id < other.id
        if op == 1:
            return self.id <= other.id
        if op == 4:
            return self.id > other.id
        if op == 5:
            return self.id >= other.id
        return NotImplemented


def from_pointer(ptr):
    """Extracts a Device object from a device pointer.

    Args:
        ptr (int): Pointer to the device memory.

    Returns:
        Device: The device whose memory the pointer refers to.

    """
    # Initialize a context to workaround a bug in CUDA 10.2+. (#3991)
    runtime._ensure_context()
    attrs = runtime.pointerGetAttributes(ptr)
    return Device(attrs.device)
