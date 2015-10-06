import atexit

import six

from cupy.cuda import cublas
from cupy.cuda import runtime


class Device(object):

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
    _cublas_handles = {}

    def __init__(self, device=None):
        if device is None:
            self.id = runtime.getDevice()
        else:
            self.id = int(device)

        self._device_stack = []

    def __int__(self):
        return self.id

    def __enter__(self):
        dev = Device()
        self._device_stack.append(dev)
        if self.id != dev.id:
            self.use()
        return self

    def __exit__(self, *args):
        self._device_stack.pop().use()

    def __repr__(self):
        return '<CUDA Device %d>' % self.id

    def use(self):
        """Makes this device current.

        If you want to switch a device temporarily, use the *with* statement.

        """
        runtime.setDevice(self.id)

    def synchronize(self):
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
        major = runtime.deviceGetAttribute(75, self.id)
        minor = runtime.deviceGetAttribute(76, self.id)
        return '%d%d' % (major, minor)

    @property
    def cublas_handle(self):
        """The cuBLAS handle for this device.

        The same handle is used for the same device even if the Device instance
        itself is different.

        """
        handle = self._cublas_handles.get(self.id, None)
        if handle is None:
            with self:
                handle = cublas.create()
                self._cublas_handles[self.id] = handle
        return handle

    def __eq__(self, other):
        """Returns True if ``other`` refers to the same device."""
        return self.id == other.id

    def __ne__(self, other):
        """Returns True if ``other`` refers to a different device."""
        return self.id != other.id


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
    for handle in six.itervalues(Device._cublas_handles):
        cublas.destroy(handle)
    Device._cublas_handles = {}
