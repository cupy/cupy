from __future__ import annotations

from cupy.cuda import Device, runtime


def _get_device_id(device):
    """Normalizes a ``device=`` argument to an integer device id.

    ``None`` maps to ``-1``, which is used as a sentinel meaning "the current
    device" (i.e. no device switch). An ``int`` is returned as is, and a
    :class:`cupy.cuda.Device` contributes its ``id``. Anything else raises
    :class:`TypeError`.
    """
    if device is None:
        return -1
    if isinstance(device, Device):
        return device.id
    # Note: bool is a subclass of int; reject it explicitly to avoid
    # treating ``True``/``False`` as device 1/0.
    if isinstance(device, int) and not isinstance(device, bool):
        return device
    raise TypeError(
        'device must be an int or cupy.cuda.Device, got '
        f'{type(device).__name__!r}')


class _DeviceGuard:
    """Lightweight scoped device switch used by array creation functions.

    Sets the current device to ``device_id`` on entry, skipping the switch
    when it already matches, and restores the previous device on exit. Unlike
    :class:`cupy.cuda.Device`, it keeps the previous device on the instance
    instead of a thread-local stack, so the common same-device path is close
    to a single ``cudaGetDevice``. A negative ``device_id`` is a no-op.
    """

    __slots__ = ('_device_id', '_prev')

    def __init__(self, device_id):
        self._device_id = device_id
        self._prev = -1

    def __enter__(self):
        dev = self._device_id
        if dev < 0:
            return self
        prev = runtime.getDevice()
        self._prev = prev
        if dev != prev:
            runtime.setDevice(dev)
        return self

    def __exit__(self, *exc):
        prev = self._prev
        if prev >= 0 and prev != self._device_id:
            runtime.setDevice(prev)


def _device_guard(device):
    """Returns a :class:`_DeviceGuard` for a ``device=`` argument."""
    return _DeviceGuard(_get_device_id(device))
