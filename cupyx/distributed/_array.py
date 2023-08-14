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

    def _get_execution_devices(self, dist_args):
        devices = set()
        for _, arg in dist_args:
            # The key of chunks is the device id
            for dev in arg._chunks:
                devices.add(dev)
        return devices

    def _get_chunk(self, i):
        return self._chunks[i]

    def _prepare_args(self, dist_args, regular_args, device):
        # Dist arrays must have chunks of compatible shapes, otherwise
        # hard error.
        # In case that they are of different, but broadcastable shapes
        # Data movement may be needed
        # Currently: Support only same shape chunks
        args = []
        c_shape = None
        for (i, arg) in dist_args:
            chunk = arg._get_chunk(device)
            args.append((i, chunk))
            if c_shape is None:
                c_shape = chunk.shape
            # TODO(ecastill) check if broadcastable, the array must have been
            # split in the same axis?
            if chunk.shape != c_shape:
                raise RuntimeError(
                    'Operating distributed arrays of different chunk sizes'
                    ' together is not supported')

        # Case of X.T and other data movement requiring cases not supported
        # TODO(ecastill) add support for operands being non distributed arrays
        # 1. Check if the regular arrays are on the specified device or
        #    peer access is enabled
        # 2. Check that their shape is compatible with the chunks
        #    distributed arrays
        # 3. Create views of this array and copy to the given device if needed
        #    so that the chunks in the distributed operate with the right slice
        if len(regular_args) > 0:
            raise RuntimeError(
                'Mix `cupy.ndarray` with distributed arrays is not currently'
                'supported')

        return args

    def _execute_kernel(self, kernel, args, kwargs):
        distributed_arrays = []
        regular_arrays = []
        for i, arg in enumerate(args):
            if isinstance(arg, _DistributedArray):
                distributed_arrays.append((i, arg))
            elif isinstance(arg, cupy.ndarray):
                regular_arrays.append((i, arg))

        # Do it for kwargs too
        for k, arg in kwargs.items():
            if isinstance(arg, _DistributedArray):
                distributed_arrays.append((k, arg))
            elif isinstance(arg, cupy.ndarray):
                regular_arrays.append((k, arg))

        args = list(args)
        devices = self._get_execution_devices(distributed_arrays)
        dev_outs = {}
        dtype = None
        for dev in devices:
            array_args = self._prepare_args(
                distributed_arrays, regular_arrays, dev)
            for (i, arg) in array_args:
                if isinstance(i, int):
                    args[i] = arg
                else:
                    kwargs[i] = arg
            with cupy.cuda.Device(dev):
                out = kernel(*args, **kwargs)
                dtype = out.dtype
                dev_outs[dev] = out

        for out in dev_outs.values():
            if not isinstance(out, cupy.ndarray):
                raise RuntimeError(
                    'kernels returning other than single array not supported')

        # Elementwise operations preserve device_mapping
        return _DistributedArray(
            self.shape, dtype, dev_outs, self._device_mapping)

    def __cupy_override_elementwise_kernel__(self, kernel, *args, **kwargs):
        # This defines a protocol to be called from elementwise kernel
        # to override some of the ops done there
        outs = self._execute_kernel(kernel, args, kwargs)
        return outs

    def asnumpy(self):
        np_array = numpy.zeros(self.shape)
        for dev, chunk_key in self._device_mapping.items():
            # Multiple writes to a single element can happen
            # This is fine since we only support replica mode now
            np_array[chunk_key] = cupy.asnumpy(self._chunks[dev])
        return np_array


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
        if isinstance(array, cupy.ndarray):
            chunk = cupy.ascontiguousarray(array[chunk_key])
        else:
            chunk = array[chunk_key]
        with cupy.cuda.Device(dev):
            cp_chunks[dev] = cupy.array(chunk)
    return _DistributedArray(array.shape, array.dtype, cp_chunks, device_mapping)

