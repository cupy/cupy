import cupy
import numpy


class _MultiDeviceDummyMemory(cupy.cuda.Memory):
    pass


class _MultiDeviceDummyPointer(cupy.cuda.MemoryPointer):
    @property
    def device(self):
        # This override is needed to assign an invalid device id
        # Since the array is not residing in a single device now
        return cupy.cuda.device.Device(-1)


class _DistributedArray(cupy.ndarray):
    # Explicitly disable indexing, view, broadcast
    def __new__(cls, shape, dtype, chunks, axis, tile_size, devices):
        # Create a dummy memptr for the array data, we will also
        # override its device methods
        # memptr is needed to avoid actual device allocation on array init
        # we need to generate other devices
        mem = _MultiDeviceDummyMemory(0)
        memptr = _MultiDeviceDummyPointer(mem, 0)
        obj = super().__new__(cls, shape, dtype, memptr=memptr)
        obj._chunks = chunks
        obj._tile_size = tile_size
        obj._mem = mem
        obj._axis = axis
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._chunks = getattr(obj, 'chunks', None)
        self._tile_size = getattr(obj, 'tile_size', None)
        self._axis = getattr(obj, 'axis', None)
        self._mem = getattr(obj, 'mem', None)

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

    def _get_execution_devices(self, dist_args):
        devices = set()
        for _, arg in dist_args:
            # The key of chunks is the device id
            for dev in arg._chunks:
                devices.add(dev)
        return devices

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

        return _DistributedArray(
            self.shape, dtype, dev_outs,
            self._axis, self._tile_size, devices)

    def __cupy_override_elementwise_kernel__(self, kernel, *args, **kwargs):
        # This defines a protocol to be called from elementwise kernel
        # to override some of the ops done there
        outs = self._execute_kernel(kernel, args, kwargs)
        return outs

    def asnumpy(self):
        # Coalesce it in a single array
        chunks = [cupy.asnumpy(c) for c in self._chunks.values()]
        return numpy.concatenate(chunks, axis=self._axis)


def _split_tiles(array, nr_splits):
    # TODO(ecastill) support arbitrary tiles shape
    axis = nr_splits.argmax()
    ind = nr_splits[axis]
    if isinstance(array, cupy.ndarray):
        arrs = [cupy.ascontiguousarray(a)
                for a in cupy.array_split(array, int(ind), int(axis))]
    else:
        arrs = numpy.array_split(array, ind, axis)
    return axis, arrs


def array(array, devices, tile_shape):
    """ Create an array that is splitted accross multiple devices in the
    same host.
    """
    if not isinstance(array, (numpy.ndarray, cupy.ndarray)):
        raise TypeError('`array` needs to be a numpy or cupy array')
    # 1. Check that array_size / tile_shape == len(devices)
    array_shape = numpy.array(array.shape, dtype=numpy.int32)
    tile_shape = numpy.array(tile_shape, dtype=numpy.int32)
    nr_splits = numpy.ceil(array_shape / tile_shape)
    if nr_splits[nr_splits > 1].size > 1:
        raise RuntimeError(
            'Currently only sharding across a single axis is allowed')

    assert len(tile_shape) == len(array_shape)

    nr_chunks = numpy.prod(nr_splits)

    if nr_chunks != len(devices):
        raise RuntimeError(
            'Array chunks must match the amount of devices')

    axis, arrs = _split_tiles(array, nr_splits)
    cp_chunks = {}
    for arr, dev in zip(arrs, devices):
        with cupy.cuda.Device(dev):
            cp_chunks[dev] = cupy.array(arr)
    return _DistributedArray(
        array_shape, array.dtype, cp_chunks, axis, tile_shape, devices)
