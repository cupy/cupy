from __future__ import annotations

from cupy import cuda


# TODO(leofang): allow explicitly wrapping a list of custom memory resources?
def cuda_core_device_memory_resource_adaptor():
    """A CuPy allocator that wraps DeviceMemoryResource from cuda.core.

    Currently, this adaptor wraps cuda.core's default memory resources. To use
    custom memory resources, attach them to the respective devices first
    (``Device.memory_resource``).
    """
    try:
        from cuda.core.experimental import system
    except ImportError:
        try:
            from cuda.core import system
        except ImportError:
            raise ModuleNotFoundError("No module named 'cuda.core'")
    devices = system.devices

    class MemoryAsyncPoolAdaptor(cuda.MemoryAsyncPool):

        __slots__ = ('mrs',)

        def __init__(self):
            MRs = [dev.memory_resource for dev in devices]
            self.mrs = MRs
            super().__init__(pool_handles=[int(mr.handle) for mr in MRs])

    return MemoryAsyncPoolAdaptor
