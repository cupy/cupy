import enum

import cupy
import numpy

from numpy.typing import ArrayLike


class _MultiDeviceDummyMemory(cupy.cuda.Memory):
    pass


class _MultiDeviceDummyPointer(cupy.cuda.MemoryPointer):
    @property
    def device(self):
        # This override is needed to assign an invalid device id
        # Since the array is not residing in a single device now
        return cupy.cuda.device.Device(-1)


class _DistributedArray(cupy.ndarray):
    def __new__(cls, shape, dtype, chunks, device_mapping):
        mem = _MultiDeviceDummyMemory(0)
        memptr = _MultiDeviceDummyPointer(mem, 0)
        obj = super().__new__(cls, shape, dtype, memptr=memptr)
        obj._chunks = chunks
        obj._device_mapping = device_mapping
        obj._mem = mem
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._chunks = getattr(obj, 'chunks', None)
        self._mem = getattr(obj, 'mem', None)


class Mode(enum.Enum):
    replica = 1


def distributed_array(
        array: ArrayLike,
        device_mapping: dict[int, tuple[slice, ...]],
        mode: Mode = Mode.replica
):
    if not isinstance(array, (numpy.ndarray, cupy.ndarray)):
        array = numpy.array(array)

    cp_chunks = {}
    for dev, chunk_key in device_mapping.items():
        with cupy.cuda.Device(dev):
            cp_chunks[dev] = cupy.array(array[chunk_key])
    return _DistributedArray(array.shape, array.dtype, cp_chunks, device_mapping)

