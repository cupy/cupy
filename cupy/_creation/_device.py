from __future__ import annotations

from cupy.cuda import Device, runtime


def _get_device_id(device):
    """Normalizes a non-``None`` ``device=`` argument to an integer id."""
    if isinstance(device, Device):
        return device.id
    # bool is an int subclass; reject it so True/False aren't devices 1/0.
    if isinstance(device, int) and not isinstance(device, bool):
        return device
    raise TypeError(
        'device must be an int or cupy.cuda.Device, got '
        f'{type(device).__name__!r}')


def _on_device(device_id, make):
    """Runs ``make()`` with ``device_id`` current, then restores the device.

    Calls cudart directly and keeps no state, following CuPy's convention of
    not using a context manager for internal device switches.
    """
    prev = runtime.getDevice()
    if device_id != prev:
        runtime.setDevice(device_id)
    try:
        return make()
    finally:
        if device_id != prev:
            runtime.setDevice(prev)
