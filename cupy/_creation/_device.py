from __future__ import annotations

from cupy.cuda import Device, runtime


def _get_device_id(device):
    """Normalizes a ``device=`` argument to an integer device id.

    ``None`` maps to ``-1`` (the current device); an ``int`` or
    :class:`cupy.cuda.Device` gives the device id. Anything else raises
    :class:`TypeError`.
    """
    if device is None:
        return -1
    if isinstance(device, Device):
        return device.id
    # bool is an int subclass; reject it so True/False aren't devices 1/0.
    if isinstance(device, int) and not isinstance(device, bool):
        return device
    raise TypeError(
        'device must be an int or cupy.cuda.Device, got '
        f'{type(device).__name__!r}')


class _DeviceGuard:
    """Lightweight scoped device switch used by array creation functions.

    Sets the current device to ``device_id`` on entry (skipping the switch when
    it already matches) and restores it on exit. A negative ``device_id`` is a
    no-op.
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
