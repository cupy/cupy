from __future__ import annotations

from cupy.cuda import runtime
from cupy.cuda.device import _normalize_device_id


def _on_device(device, func, *args, **kwargs):
    """Calls ``func(*args, **kwargs)`` on ``device``, then restores the device.

    Creation functions take this branch only when ``device is not None`` and
    re-enter themselves with the argument dropped, so the default path costs
    one ``is not None`` test. Passing ``func`` and its arguments rather than a
    closure keeps the callers' locals out of cells, which would otherwise slow
    down the ``device=None`` path as well.

    The whole call runs on ``device``, not only the allocation: the kernels
    that initialize the array (``fill``, ``copyto``, ...) run on the current
    device and reject arrays that live elsewhere.

    Calls cudart directly and keeps no state, following CuPy's convention of
    not wrapping internal device switches in a context manager.
    """
    dev = _normalize_device_id(device)
    prev = runtime.getDevice()
    if dev != prev:
        runtime.setDevice(dev)
    try:
        return func(*args, **kwargs)
    finally:
        if dev != prev:
            runtime.setDevice(prev)
