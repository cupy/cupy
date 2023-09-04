import dataclasses
from typing import Any, Callable, Optional, TypeVar

from cupy.cuda import nccl, Device, Stream

import cupy
import numpy

from numpy.typing import ArrayLike
from cupy.typing import NDArray
from cupy.typing._types import _ScalarType_co
from cupyx.distributed._nccl_comm import _nccl_dtypes


def _extgcd(a: int, b: int) -> tuple[int, int]:
    """Return (g, x) with g = gcd(a, b), ax + by = g - ax.
    a, b > 0 is assumed."""
    # c - ax - by = 0  ...  (1)
    # d - au - bv = 0  ...  (2)
    c, d = a, b
    x, u = 1, 0
    # y, v = 0, 1

    # Apply Euclid's algorithm to (c, d)
    while d:
        r = c // d
        # (1), (2) = (2), (1) - (2) * r
        c, d = d, c - d * r
        x, u = u, x - u * r
        # y, v = v, y - u * r

    return c, x


def _slice_intersection(a: slice, b: slice, length: int) -> Optional[slice]:
    """Return the intersection of slice a and b. None if they are disjoint."""
    a_start, a_stop, a_step = a.indices(length)
    b_start, b_stop, b_step = b.indices(length)

    # a_step * x + b_step * y == g  ...  (1)
    g, x = _extgcd(a_step, b_step)
    if (b_start - a_start) % g != 0:
        return None

    # c is the intersection of a, b
    # c_step == lcm(a_step, b_step)
    c_step = a_step // g * b_step

    # Multiply (1) by (b_start - a_start) // g
    # ==> a_step * a_skip - b_step * b_skip == b_start - a_start
    #     a_start + a_step * a_skip == b_start + b_step * b_skip
    a_skip = x * ((b_start - a_start) // g) % (c_step // a_step)
    c_start = a_start + a_step * a_skip
    if c_start < b_start:
        c_start += ((b_start - c_start - 1) // c_step + 1) * c_step

    c_stop = min(a_stop, b_stop)
    if c_start < c_stop:
        return slice(c_start, c_stop, c_step)
    else:
        return None


def _index_for_subslice(a: slice, sub: slice, length: int) -> slice:
    """Return slice c such that array[a][c] == array[sub].
    sub should be contained in a."""
    a_start, a_stop, a_step = a.indices(length)
    sub_start, sub_stop, sub_step = sub.indices(length)

    c_start = (sub_start - a_start) // a_step
    # a_start + a_step * (c_stop - 1) < sub_stop
    c_stop = (sub_stop - a_start - 1) // a_step + 1
    c_step = sub_step // a_step

    return slice(c_start, c_stop, c_step)


def _index_intersection(
        a_idx: tuple[slice, ...], b_idx: tuple[slice, ...],
        shape: tuple[int, ...]) -> Optional[tuple[slice, ...]]:
    """Return None if empty."""
    ndim = len(shape)
    assert len(a_idx) == len(b_idx) == ndim
    result = tuple(_slice_intersection(a, b, length)
                   for a, b, length in zip(a_idx, b_idx, shape))
    if None in result:
        return None
    else:
        return result


def _index_for_subindex(
        a_idx: tuple[slice, ...], sub_idx: tuple[slice, ...],
        shape: tuple[int, ...]) -> tuple[slice, ...]:
    ndim = len(shape)
    assert len(a_idx) == len(sub_idx) == ndim

    return tuple(_index_for_subslice(a, sub, length)
                   for a, sub, length in zip(a_idx, sub_idx, shape))


# Temporary helper function.
# Should be removed after implementing indexing
def _shape_after_indexing(
        outer_shape: tuple[int, ...],
        idx: tuple[slice, ...]) -> tuple[int, ...]:
    shape = list(outer_shape)
    for i in range(len(idx)):
        start, stop, step = idx[i].indices(shape[i])
        shape[i] = (stop - start - 1) // step + 1
    return tuple(shape)


def _convert_chunk_idx_to_slices(
        shape: tuple[int, ...], idx: Any) -> tuple[slice, ...]:
    """Convert idx to type tuple[slice, ...] with all nonnegative indices and
    length == ndim. Raise if empty or invalid.

    Negative slice steps are not allowed, because this function is for
    representing chunks, e.g. the indices in device_mapping."""

    if not isinstance(idx, tuple):
        idx = idx,

    ndim = len(shape)
    if len(idx) > ndim:
        raise IndexError(
            'too many indices for array:'
            f' array is {ndim}-dimensional, but {len(idx)} were indexed')
    idx = idx + (slice(None),) * (ndim - len(idx))

    new_idx = []
    for i in range(ndim):
        if isinstance(idx[i], int):
            if idx[i] >= shape[i]:
                raise IndexError(
                    f'Index {idx[i]} is out of bounds'
                    f' for axis {i} with size {shape[i]}')
            new_idx.append(slice(idx[i], idx[i] + 1))
        elif isinstance(idx[i], slice):
            start, stop, step = idx[i].indices(shape[i])
            if step == 0:
                raise ValueError('Slice step must be nonzero')
            if step < 0:
                raise ValueError(
                    'The indices for a chunk cannot have negative slice steps.')
            if start == stop:
                raise ValueError(f'The index is empty on axis {i}')
            new_idx.append(slice(start, stop, step))
        else:
            raise ValueError(f'Invalid index on axis {i}')

    return tuple(new_idx)


# Copied from cupyx/distributed/_nccl_comm.py
def _get_nccl_dtype_and_count(array, count=None):
    dtype = array.dtype.char
    if dtype not in _nccl_dtypes:
        raise TypeError(f'Unknown dtype {array.dtype} for NCCL')
    nccl_dtype = _nccl_dtypes[dtype]
    if count is None:
        count = array.size
    if dtype in 'FD':
        return nccl_dtype, 2 * count
    return nccl_dtype, count


class _MultiDeviceDummyMemory(cupy.cuda.Memory):
    pass


class _MultiDeviceDummyPointer(cupy.cuda.MemoryPointer):
    @property
    def device(self):
        # This override is needed to assign an invalid device id
        # Since the array is not residing in a single device now
        return Device(-1)


def _min_value_of(dtype):
    if dtype.kind in 'biu':
        return cupy.iinfo(dtype).min
    elif dtype.kind in 'f':
        return -cupy.inf


def _max_value_of(dtype):
    if dtype.kind in 'biu':
        return cupy.iinfo(dtype).max
    elif dtype.kind in 'f':
        return cupy.inf


def _zero_value_of(dtype):
    return dtype.type(0)


def _one_value_of(dtype):
    return dtype.type(1)


@dataclasses.dataclass
class _OpMode:
    func: cupy.ufunc
    numpy_func: numpy.ufunc
    idempotent: bool
    identity_of: Callable[[numpy.dtype], _ScalarType_co]


_Mode = Optional[_OpMode]


_REPLICA_MODE: _Mode = None


_MODES = {
    'replica': _REPLICA_MODE,
    'min': _OpMode(cupy.minimum, numpy.minimum, True, _max_value_of),
    'max': _OpMode(cupy.maximum, numpy.maximum, True, _min_value_of),
    'sum': _OpMode(cupy.add, numpy.add, False, _zero_value_of),
    'prod': _OpMode(cupy.multiply, numpy.multiply, False, _one_value_of),
}


@dataclasses.dataclass
class _ControlledData:
    """ND-array managed by a stream."""
    data: NDArray
    stream: Stream = dataclasses.field(default_factory=Stream)


# Overwrite in replica mode, apply in op mode
_PartialUpdate = tuple[_ControlledData, tuple[slice, ...]]


def _copy_controlled_data(cdata: _ControlledData):
    with cdata.data.device:
        copy = Stream()
        copy.wait_event(cdata.stream.record())
        with copy:
            return _ControlledData(cdata.data.copy(), copy)


class _DistributedArray(cupy.ndarray):
    # Array on the devices and streams that transfer data only within their
    # corresponding device
    _chunks: dict[int, _ControlledData]
    _device_mapping: dict[int, tuple[slice, ...]]
    _mode: _Mode
    # Buffers for transfer from other devices
    _updates: dict[int, list[_PartialUpdate]]
    _comms: dict[int, nccl.NcclCommunicator]
    _mem: cupy.cuda.Memory

    def __new__(
            cls, shape, dtype, chunks, device_mapping, mode=_REPLICA_MODE,
            comms=None, updates=None):
        mem = _MultiDeviceDummyMemory(0)
        memptr = _MultiDeviceDummyPointer(mem, 0)
        obj = super().__new__(cls, shape, dtype, memptr=memptr)
        obj._chunks = chunks
        obj._device_mapping = device_mapping
        obj._mode = mode
        if comms:
            obj._comms = comms
        elif nccl.available:
            comms_list = nccl.NcclCommunicator.initAll(list(device_mapping))
            obj._comms = {dev: comm
                        for dev, comm in zip(device_mapping, comms_list)}
        else:
            # TODO: support environments where NCCL is unavailable
            raise RuntimeError('NCCL is unavailable')
        if updates:
            obj._updates = updates
        else:
            obj._updates = {dev: [] for dev in device_mapping}
        obj._mem = mem
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._chunks = getattr(obj, '_chunks', None)
        self._device_mapping = getattr(obj, '_device_mapping', None)
        self._mode = getattr(obj, '_mode', None)
        self._updates = getattr(obj, '_updates', None)
        self._comms = getattr(obj, '_comms', None)
        self._mem = getattr(obj, '_mem', None)

    @property
    def mode(self):
        return self._mode

    @property
    def device_mapping(self):
        return self._device_mapping

    def _get_execution_devices(self, dist_args):
        devices = set()
        for _, arg in dist_args:
            for dev in arg._chunks:
                devices.add(dev)
        return devices

    def _get_chunk(self, i):
        return self._chunks[i]

    def _collect_chunk(self, mode: _Mode, chunk: _ControlledData,
                       updates: list[_PartialUpdate]) -> None:
        """Apply all updates onto `chunk` in-place."""
        with chunk.data.device:
            for new_data, idx in updates:
                chunk.stream.wait_event(new_data.stream.record())

                with chunk.stream:
                    if mode is _REPLICA_MODE:
                        chunk.data[idx] = new_data.data
                    else:
                        mode.func(
                            chunk.data[idx], new_data.data, chunk.data[idx])
        updates.clear()

    def _wait_all_transfer(self):
        """Block until all inner updates are done."""

        transfer_events = []

        for dev, chunk in self._chunks.items():
            self._collect_chunk(
                self._mode, self._chunks[dev], self._updates[dev])
            with Device(dev):
                transfer_events.append(chunk.stream.record())

        for e in transfer_events:
            e.synchronize()

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
                c_shape = chunk.data.shape
            # TODO(ecastill) check if broadcastable, the array must have been
            # split in the same axis?
            if chunk.data.shape != c_shape:
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
                'Mix `cupy.ndarray` with distributed arrays is currently not'
                'supported')

        return args

    def _prepare_updates(self, dist_args, dev):
        index = None
        updates = []
        at_most_one_update = True
        for i, arg in dist_args:
            if arg._updates[dev]:
                if updates:
                    at_most_one_update = False
                    break
                index = i
                updates = arg._updates[dev]

        # If there is at most one array with partial updates, we return them
        # and apply the element-wise kernel without actually propagating
        # those updates. Otherwise we propagate them beforehand.
        # TODO check if it really gives any speedup
        if at_most_one_update:
            return index, updates

        for i, arg in dist_args:
            self._collect_chunk(
                _REPLICA_MODE, arg._chunks[dev], arg._updates[dev])
        return None, []

    def _execute_kernel(self, kernel, args, kwargs) -> Any:
        distributed_arrays: list[tuple[int | str, '_DistributedArray']] = []
        regular_arrays: list[tuple[int | str, NDArray]] = []
        for i, arg in enumerate(args):
            if isinstance(arg, _DistributedArray):
                distributed_arrays.append((i, arg.to_replica_mode()))
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
        new_dtype = None
        new_chunks = {}
        new_updates = {dev: [] for dev in self._device_mapping}
        for dev in devices:
            with Device(dev):
                array_args = self._prepare_args(
                    distributed_arrays, regular_arrays, dev)

                apply_kernel = Stream()
                for i, arg in array_args:
                    apply_kernel.wait_event(arg.stream.record())
                    if isinstance(i, int):
                        args[i] = arg.data
                    else:
                        kwargs[i] = arg.data

                args_ready = apply_kernel.record()
                with apply_kernel:
                    chunk = kernel(*args, **kwargs)

                new_dtype = chunk.dtype
                new_chunks[dev] = _ControlledData(chunk, apply_kernel)

                incoming_index, updates = self._prepare_updates(
                    distributed_arrays, dev)
                if len(updates) == 0:
                    continue

                args_slice = [None] * len(args)
                kwargs_slice = {}
                for update, idx in updates:
                    apply_kernel_to_update = Stream()
                    apply_kernel_to_update.wait_event(args_ready)
                    for i, arg in enumerate(args):
                        args_slice[i] = arg[idx]
                    for i, arg in kwargs.items():
                        kwargs_slice[i] = arg[idx]

                    if isinstance(incoming_index, int):
                        args_slice[incoming_index] = update.data
                    else:
                        kwargs_slice[incoming_index] = update.data

                    with apply_kernel_to_update:
                        new_data = kernel(*args_slice, **kwargs_slice)

                    new_updates[dev].append(
                        (_ControlledData(new_data, apply_kernel_to_update),
                         idx))

        for out in new_chunks.values():
            if not isinstance(out.data, cupy.ndarray):
                raise RuntimeError(
                    'Kernels returning other than signle array not supported')

        return _DistributedArray(
            self.shape, new_dtype, new_chunks, self._device_mapping,
            _REPLICA_MODE, self._comms, new_updates)

    def __cupy_override_elementwise_kernel__(self, kernel, *args, **kwargs):
        # This defines a protocol to be called from elementwise kernel
        # to override some of the ops done there
        outs = self._execute_kernel(kernel, args, kwargs)
        return outs

    def _transfer_async(
            self, src_chunk: _ControlledData, dst_dev: int) -> _ControlledData:
        src_dev = src_chunk.data.device.id

        with Device(src_dev):
            src_stream = Stream()
            src_stream.wait_event(src_chunk.stream.record())
            with src_stream:
                src_array = cupy.ascontiguousarray(src_chunk.data)
        with Device(dst_dev):
            dst_stream = Stream()
            with dst_stream:
                dst_buf = cupy.empty_like(src_array)

        dtype, count = _get_nccl_dtype_and_count(src_array)
        nccl.groupStart()

        with Device(src_dev):
            self._comms[src_dev].send(
                src_array.data.ptr, count, dtype, dst_dev, src_stream.ptr)

        with Device(dst_dev):
            self._comms[dst_dev].recv(
                dst_buf.data.ptr, count, dtype, src_dev, dst_stream.ptr)

            nccl.groupEnd()
            return _ControlledData(dst_buf, dst_stream)

    def _copy_to(
            self, src_array: NDArray, dst_array: NDArray) -> None:
        dst_dev = dst_array.device.id
        with Device(dst_dev):
            dst_array[:] = src_array.copy()

    def _apply_and_update_chunks(
            self, op_mode, shape, updates,  # TODO only receive updates[dst_dev]
            src_chunk: _ControlledData, src_idx,
            dst_chunk: _ControlledData, dst_idx):
        """Apply `src_chunk` onto `dst_chunk` in `op_mode`.
        There must not be any undone partial update to src_chunk."""
        src_dev = src_chunk.data.device.id
        dst_dev = dst_chunk.data.device.id

        intersection = _index_intersection(src_idx, dst_idx, shape)
        if intersection is None:
            return
        src_new_idx = _index_for_subindex(src_idx, intersection, shape)
        dst_new_idx = _index_for_subindex(dst_idx, intersection, shape)

        data_to_transfer = _ControlledData(
            src_chunk.data[src_new_idx], src_chunk.stream)

        update = self._transfer_async(data_to_transfer, dst_dev)

        updates[dst_dev].append((update, dst_new_idx))

        if not op_mode.idempotent:
            dtype = src_chunk.data.dtype
            with Device(src_dev):
                with src_chunk.stream:
                    src_chunk.data[src_new_idx] = op_mode.identity_of(dtype)

    def _all_reduce_intersections(
            self, op_mode, shape, chunks, updates, device_mapping):
        chunks_list = list(chunks.items())

        for i in range(len(chunks_list)):
            src_dev, src_chunk = chunks_list[i]
            src_idx = device_mapping[src_dev]
            self._collect_chunk(op_mode, src_chunk, updates[src_dev])

            for j in range(i + 1, len(chunks_list)):
                dst_dev, dst_chunk = chunks_list[j]
                dst_idx = device_mapping[dst_dev]

                self._apply_and_update_chunks(
                    op_mode, shape, updates,
                    src_chunk, src_idx, dst_chunk, dst_idx)

        for j in range(len(chunks_list) - 1, -1, -1):
            src_dev, src_chunk = chunks_list[j]
            src_idx = device_mapping[src_dev]
            self._collect_chunk(_REPLICA_MODE, src_chunk, updates[src_dev])

            for i in range(j):
                dst_dev, dst_chunk = chunks_list[i]
                dst_idx = device_mapping[dst_dev]

                self._copy_on_intersection(
                    shape, updates, src_chunk, src_idx, dst_chunk, dst_idx)

    def _copy_on_intersection(
            self, shape: tuple[int, ...],
            updates: dict[int, list[_PartialUpdate]],
            src_chunk: _ControlledData, src_idx: tuple[slice, ...],
            dst_chunk: _ControlledData, dst_idx: tuple[slice, ...]) -> None:
        # intersection == src_chunk[src_new_idx] == dst_chunk[dst_new_idx]
        intersection = _index_intersection(src_idx, dst_idx, shape)
        if intersection is None:
            return

        dst_dev = dst_chunk.data.device.id
        src_new_idx = _index_for_subindex(src_idx, intersection, shape)
        dst_new_idx = _index_for_subindex(dst_idx, intersection, shape)

        src_partial_chunk = _ControlledData(
            src_chunk.data[src_new_idx], src_chunk.stream)
        update = self._transfer_async(src_partial_chunk, dst_dev)
        updates[dst_dev].append((update, dst_new_idx))

    def _set_identity_on_intersection(
            self, shape: tuple[int, ...], identity,
            a_chunk: _ControlledData, a_idx: tuple[slice, ...],
            b_idx: tuple[slice, ...]) -> None:
        intersection = _index_intersection(a_idx, b_idx, shape)
        if intersection is None:
            return
        a_new_idx = _index_for_subindex(a_idx, intersection, shape)
        with Device(a_chunk.data.device.id):
            with a_chunk.stream:
                a_chunk.data[a_new_idx] = identity

    def _set_identity_on_ignored_entries(
            self, identity, dev,
            chunk: _ControlledData, updates: list[_PartialUpdate]) -> None:
        with Device(dev):
            with chunk.stream:
                for _, idx in updates:
                    chunk.data[idx] = identity

    def __cupy_override_reduction_kernel__(
            self, kernel, axis, dtype, out, keepdims):
        if out is not None:
            raise RuntimeError('Argument `out` is not supported')
        if keepdims:
            raise RuntimeError('`keepdims` is not supported')

        if kernel.name == 'cupy_max':
            mode = _MODES['max']
            chunks, updates = self._replica_mode_chunks_and_updates()
        elif kernel.name == 'cupy_min':
            mode = _MODES['min']
            chunks, updates = self._replica_mode_chunks_and_updates()
        elif kernel.name == 'cupy_sum':
            mode = _MODES['sum']
            chunks, updates = self._op_mode_chunks_and_updates(mode)
        elif kernel.name == 'cupy_prod':
            mode = _MODES['prod']
            chunks, updates = self._op_mode_chunks_and_updates(mode)
        else:
            raise RuntimeError(f'Unsupported kernel: {kernel.name}')

        identity = mode.identity_of(self.dtype)
        for dev, chunk in chunks.items():
            self._set_identity_on_ignored_entries(
                identity, dev, chunk, updates[dev])

        shape = self.shape[:axis] + self.shape[axis+1:]
        new_dtype = None
        new_chunks = {}
        new_device_mapping = {}
        new_updates = {dev: [] for dev in self._device_mapping}

        for dev, chunk in chunks.items():
            idx = self._device_mapping[dev]
            with Device(dev):
                apply_kernel = Stream()
                apply_kernel.wait_event(chunk.stream.record())
                with apply_kernel:
                    new_data = kernel(chunk.data, axis=axis, dtype=dtype)
                new_chunks[dev] = _ControlledData(new_data, apply_kernel)

            new_dtype = new_chunks[dev].data.dtype
            new_device_mapping[dev] = idx[:axis] + idx[axis+1:]

        for dev, upds in updates.items():
            with Device(dev):
                for update, idx in upds:
                    stream = Stream()
                    stream.wait_event(update.stream.record())
                    with stream:
                        new_data = kernel(update.data, axis=axis, dtype=dtype)
                    new_idx = idx[:axis] + idx[axis+1:]
                    new_updates[dev].append(
                        (_ControlledData(new_data, stream), new_idx))

        return _DistributedArray(
            shape, new_dtype, new_chunks, new_device_mapping, mode, self._comms,
            new_updates)

    def _copy_chunks_and_updates(self):
        chunks = {dev: _copy_controlled_data(chunk)
                  for dev, chunk in self._chunks.items()}
        updates = {dev: list(upds) for dev, upds in self._updates.items()}
        return chunks, updates

    def _replica_mode_chunks_and_updates(
            self
        ) -> tuple[dict[int, _ControlledData], dict[int, list[_PartialUpdate]]]:
        """Make copies of the chunks and their updates into the replica mode."""
        if self._mode is _REPLICA_MODE:
            return self._copy_chunks_and_updates()

        chunks, updates = self._copy_chunks_and_updates()
        self._all_reduce_intersections(
            self._mode, self.shape, chunks, updates, self._device_mapping)
        return chunks, updates

    def _op_mode_chunks_and_updates(
            self, op_mode
        ) -> tuple[dict[int, NDArray], dict[int, list[_PartialUpdate]]]:
        """Make copies of the chunks and their updates into the given mode."""
        if self._mode is op_mode:
            return self._copy_chunks_and_updates()

        chunks, updates = self._replica_mode_chunks_and_updates()
        for dev in chunks:
            self._collect_chunk(_REPLICA_MODE, chunks[dev], updates[dev])

        chunks_list = list(chunks.items())
        identity = op_mode.identity_of(self.dtype)
        new_chunks = {}
        # TODO: Parallelize
        for i in range(len(chunks_list)):
            a_dev, a_chunk = chunks_list[i]
            a_idx = self._device_mapping[a_dev]

            for j in range(i + 1, len(chunks_list)):
                b_dev, _ = chunks_list[j]
                b_idx = self._device_mapping[b_dev]

                self._set_identity_on_intersection(
                    self.shape, identity, a_chunk, a_idx, b_idx)

            new_chunks[a_dev] = a_chunk

        return new_chunks, updates

    def to_replica_mode(self) -> '_DistributedArray':
        """Does not recessarily copy."""
        if self._mode is _REPLICA_MODE:
            return self
        else:
            chunks, updates = self._replica_mode_chunks_and_updates()
            return _DistributedArray(
                self.shape, self.dtype, chunks, self._device_mapping,
                _REPLICA_MODE, self._comms, updates)

    def change_mode(self, mode: str) -> '_DistributedArray':
        if _MODES[mode] is self._mode:
            return self

        if mode == 'replica':
            chunks, updates = self._replica_mode_chunks_and_updates()
        else:
            chunks, updates = self._op_mode_chunks_and_updates(_MODES[mode])
        return _DistributedArray(
            self.shape, self.dtype, chunks, self._device_mapping, _MODES[mode],
            self._comms, updates)

    def reshard(
        self, device_mapping: dict[int, Any]
    ) -> '_DistributedArray':
        for dev in device_mapping:
            if dev not in self._comms:
                raise RuntimeError(
                    f'A communicator for device {dev} is not prepared.')

        old_mapping = self._device_mapping
        new_mapping = {dev: _convert_chunk_idx_to_slices(self.shape, idx)
                       for dev, idx in device_mapping.items()}

        old_chunks = self._chunks
        new_chunks = {}

        new_updates = {dev: [] for dev in new_mapping}
        if self._mode is not None:
            identity = self._mode.identity_of(self.dtype)
        for dst_dev, dst_idx in new_mapping.items():
            with Device(dst_dev):
                dst_shape = _shape_after_indexing(self.shape, dst_idx)
                stream = Stream()
                with stream:
                    if self._mode is _REPLICA_MODE:
                        data = cupy.empty(dst_shape, self.dtype)
                    else:
                        data = cupy.full(dst_shape, identity, self.dtype)
                new_chunks[dst_dev] = _ControlledData(data, stream)

        for src_dev, src_idx in old_mapping.items():
            src_chunk = self._chunks[src_dev]
            self._collect_chunk(
                self._mode, src_chunk, self._updates[src_dev])
            if self._mode is not None:
                src_chunk = _copy_controlled_data(src_chunk)

            for dst_dev, dst_idx in new_mapping.items():
                dst_chunk = new_chunks[dst_dev]

                if self._mode is _REPLICA_MODE:
                    self._copy_on_intersection(
                        self.shape, new_updates,
                        src_chunk, src_idx, dst_chunk, dst_idx)
                else:
                    self._apply_and_update_chunks(
                        self._mode, self.shape, new_updates,
                        src_chunk, src_idx, dst_chunk, dst_idx)

        return _DistributedArray(
            self.shape, self.dtype, new_chunks, new_mapping, self._mode,
            self._comms, new_updates)

    def asnumpy(self):
        for dev, chunk in self._chunks.items():
            self._collect_chunk(self._mode, chunk, self._updates[dev])

        if self._mode is _REPLICA_MODE:
            np_array = numpy.empty(self.shape, dtype=self.dtype)
        else:
            identity = self._mode.identity_of(self.dtype)
            np_array = numpy.full(self.shape, identity, self.dtype)

        for dev, chunk_idx in self._device_mapping.items():
            self._chunks[dev].stream.synchronize()
            if self._mode is _REPLICA_MODE:
                np_array[chunk_idx] = cupy.asnumpy(self._chunks[dev].data)
            else:
                self._mode.numpy_func(
                    np_array[chunk_idx], cupy.asnumpy(self._chunks[dev].data),
                    np_array[chunk_idx])

        return np_array


def distributed_array(
        array: ArrayLike,
        device_mapping: dict[int, Any],
        mode: str = 'replica',
        comms: Optional[dict[int, nccl.NcclCommunicator]] = None):
    if mode not in _MODES:
        raise RuntimeError(f'`mode` must be one of {list(_MODES)}')
    mode_obj = _MODES[mode]

    if not isinstance(array, (numpy.ndarray, cupy.ndarray)):
        array = numpy.array(array)
    elif mode != 'replica':
        array = array.copy()

    device_mapping = {dev: _convert_chunk_idx_to_slices(array.shape, idx)
                      for dev, idx in device_mapping.items()}


    cp_chunks = {}
    for dev, chunk_idx in device_mapping.items():
        if isinstance(array, cupy.ndarray):
            chunk = cupy.ascontiguousarray(array[chunk_idx])
        else:
            chunk = array[chunk_idx]
        with Device(dev):
            stream = Stream()
            with stream:
                cp_chunks[dev] = _ControlledData(cupy.array(chunk), stream)
        if mode_obj is not None and not mode_obj.idempotent:
            array[chunk_idx] = mode_obj.identity_of(array.dtype)
    return _DistributedArray(
        array.shape, array.dtype, cp_chunks, device_mapping, mode_obj, comms)

