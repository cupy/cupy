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


cdef class _ThreadLocalStack:
    cdef list _devices

    def __init__(self):
        self._devices = [0]

    @staticmethod
    cdef _ThreadLocalStack get():
        try:
            stack = _thread_local._device_stack
        except AttributeError:
            stack = _ThreadLocalStack()
            _thread_local._device_stack = stack
        return <_ThreadLocalStack>stack

    cdef void push_device(self, int device_id) except *:
        self._devices.append(device_id)

    cdef int pop_device(self) except -1:
        self._devices.pop()
        return <int>self._devices[-1]


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


cdef bint _enable_peer_access(int device, int peer) except -1:
    """Enable accessing memory allocated on `peer` from `device`."""
    if device == peer:
        return True

    cdef int can_access = runtime.deviceCanAccessPeer(device, peer)
    if can_access == 0:
        return False

    cdef int current = runtime.getDevice()
    runtime.setDevice(device)
    try:
        # Note: external libraries may disable the peer access, so we need to
        # call this everytime. See #5496.
        runtime._deviceEnsurePeerAccess(peer)
    finally:
        runtime.setDevice(current)
    return True


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
        # N.B. for maintainers: do not use this context manager in CuPy
        # codebase. See #5943 and #5963.
        if self.id != runtime.getDevice():
            runtime.setDevice(self.id)
        _ThreadLocalStack.get().push_device(self.id)
        return self

    def __exit__(self, *args):
        cdef int prev_device = _ThreadLocalStack.get().pop_device()
        if prev_device != runtime.getDevice():
            runtime.setDevice(prev_device)

    def __repr__(self):
        return '<CUDA Device %d>' % self.id

    cpdef use(self):
        """Makes this device current.

        In general, usage of this method is discouraged. Instead use the Device
        object as a context manager (*with* statement) to switch the current
        device for the specified scope.

        Note that the mixed use of this method and *with* statement may cause
        surprising results in some cases:

        .. testcode::

            dev0 = cupy.cuda.Device(0)
            dev1 = cupy.cuda.Device(1)

            dev1.use()
            with dev0:
                dev1.use()
                with dev0:
                    pass
                # The current device remains 0.
                # Notice that the current device at the time of entering the
                # context is not recalled when exiting a context.
            # The current device still remains 0.

        """
        runtime.setDevice(self.id)
        return self

    cpdef synchronize(self):
        """Synchronizes the current thread to the device."""
        syncdetect._declare_synchronize()
        prev_device = runtime.getDevice()
        try:
            runtime.setDevice(self.id)
            runtime.deviceSynchronize()
        finally:
            runtime.setDevice(prev_device)

    @property
    def compute_capability(self):
        """Compute capability of this device.

        The capability is represented by a string containing the major index
        and the minor index. For example, compute capability 3.5 is represented
        by the string '35'.

        """
        if self.id in _compute_capabilities:
            return _compute_capabilities[self.id]
        prev_device = runtime.getDevice()
        try:
            runtime.setDevice(self.id)
            major = runtime.deviceGetAttribute(
                runtime.deviceAttributeComputeCapabilityMajor, self.id)
            minor = runtime.deviceGetAttribute(
                runtime.deviceAttributeComputeCapabilityMinor, self.id)
            cc = '%d%d' % (major, minor)
            _compute_capabilities[self.id] = cc
            return cc
        finally:
            runtime.setDevice(prev_device)

    def _get_handle(self, name, create_func, destroy_func):
        handles = getattr(_thread_local, name, None)
        if handles is None:
            handles = {}
            setattr(_thread_local, name, handles)
        handle = handles.get(self.id, None)
        if handle is not None:
            return handle.handle
        prev_device = runtime.getDevice()
        try:
            runtime.setDevice(self.id)
            handle = create_func()
            handles[self.id] = Handle(handle, destroy_func)
            return handle
        finally:
            runtime.setDevice(prev_device)

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
        prev_device = runtime.getDevice()
        try:
            runtime.setDevice(self.id)
            return runtime.memGetInfo()
        finally:
            runtime.setDevice(prev_device)

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
