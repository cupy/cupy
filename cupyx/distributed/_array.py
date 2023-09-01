from dataclasses import dataclass
from typing import Any, Optional

from cupy.cuda import nccl, Stream, Device

import cupy
import numpy

from numpy.typing import ArrayLike
from cupy.typing import NDArray
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


@dataclass
class OpMode:
    """None as Mode represents replica mode.
    func(x, x) == x must hold if identity is None."""
    func: cupy.ufunc
    numpy_func: numpy.ufunc
    identity: Any


Mode = Optional[OpMode]


_max_mode = OpMode(cupy.maximum, numpy.maximum, None)
_min_mode = OpMode(cupy.minimum, numpy.minimum, None)
_sum_mode = OpMode(cupy.add, numpy.add, 0)
_prod_mode = OpMode(cupy.multiply, numpy.multiply, 1)
_replica_mode: Mode = None


@dataclass
class IncomingData:
    stream: Stream
    data: NDArray


def _min_value(dtype):
    if dtype.kind in 'biu':
        return cupy.iinfo(dtype).min
    elif dtype.kind in 'f':
        return -cupy.inf


def _max_value(dtype):
    if dtype.kind in 'biu':
        return cupy.iinfo(dtype).max
    elif dtype.kind in 'f':
        return cupy.inf


class _DistributedArray(cupy.ndarray):
    _chunks: dict[int, NDArray]
    # Values of _device_mapping must have lenth == ndim
    _device_mapping: dict[int, tuple[slice, ...]]
    _mode: Mode
    _incomings: dict[int, list[tuple[IncomingData, Any]]]
    _comms: dict[int, nccl.NcclCommunicator]

    def __new__(
            cls, shape, dtype, chunks, device_mapping, mode,
            comms=None, incomings=None):
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
            obj._comms = None
        if incomings:
            obj._incomings = incomings
        else:
            obj._incomings = {dev: [] for dev in device_mapping}
        obj._mem = mem
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._chunks = getattr(obj, 'chunks', None)
        self._mem = getattr(obj, 'mem', None)

    @property
    def mode(self):
        return self._mode

    @property
    def device_mapping(self):
        return self._device_mapping

    def _get_execution_devices(self, dist_args):
        devices = set()
        for _, arg in dist_args:
            # The key of chunks is the device id
            for dev in arg._chunks:
                devices.add(dev)
        return devices

    def _get_chunk(self, i):
        return self._chunks[i]

    def _collect_chunk(self, mode, chunk, incomings) -> Stream:
        """Returns a stream that collects all incoming data onto `chunk`."""
        dev = chunk.device.id
        with Device(dev):
            waiter = Stream()
            copy_done = []
            for incoming, idx in incomings:
                with incoming.stream:
                    if mode is _replica_mode:
                        chunk[idx] = incoming.data
                    else:
                        mode.func(chunk[idx], incoming.data, chunk[idx])
                copy_done.append(incoming.stream.record())

            for e in copy_done:
                waiter.wait_event(e)

            return waiter

    def _wait_all_transfer(self):
        events = []
        for dev in self.device_mapping:
            if not self._incomings[dev]:
                continue
            collector = self._collect_chunk(
                self._mode, self._chunks[dev], self._incomings[dev])
            self._incomings[dev].clear()
            with Device(dev):
                events.append(collector.record())

        for e in events:
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
                'Mix `cupy.ndarray` with distributed arrays is currently not'
                'supported')

        return args

    def _prepare_incomings(self, dist_args, dev):
        index = None
        incomings = []
        for i, arg in dist_args:
            if arg._incomings[dev]:
                if incomings:
                    raise RuntimeError(
                        'Operationg an element-wise kernel on distributed'
                        ' arrays with possibly unfinished data transfer on two'
                        ' of them is currently not supported')
                index = i
                incomings = arg._incomings[dev]
        return index, incomings

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
        new_incomings = {dev: [] for dev in self._device_mapping}
        for dev in devices:
            with Device(dev):
                array_args = self._prepare_args(
                    distributed_arrays, regular_arrays, dev)
                for i, arg in array_args:
                    if isinstance(i, int):
                        args[i] = arg
                    else:
                        kwargs[i] = arg

                out = kernel(*args, **kwargs)
                new_dtype = out.dtype
                new_chunks[dev] = out

                incoming_index, incomings = self._prepare_incomings(
                    distributed_arrays, dev)
                if not incomings:
                    continue

                args_slice = [None] * len(args)
                kwargs_slice = {}
                for incoming, idx in incomings:
                    for i, arg in enumerate(args):
                        args_slice[i] = arg[idx]
                    for i, arg in kwargs.items():
                        kwargs_slice[i] = arg[idx]

                    if isinstance(incoming_index, int):
                        args_slice[incoming_index] = incoming.data
                    else:
                        kwargs_slice[incoming_index] = incoming.data

                    with incoming.stream:
                        new_data = kernel(*args_slice, **kwargs_slice)
                    new_incomings[dev].append(
                        (IncomingData(incoming.stream, new_data), idx))

        for out in new_chunks.values():
            if not isinstance(out, cupy.ndarray):
                raise RuntimeError(
                    'Kernels returning other than signle array not supported')

        return _DistributedArray(
            self.shape, new_dtype, new_chunks, self._device_mapping,
            _replica_mode, self._comms, new_incomings)

    def __cupy_override_elementwise_kernel__(self, kernel, *args, **kwargs):
        # This defines a protocol to be called from elementwise kernel
        # to override some of the ops done there
        outs = self._execute_kernel(kernel, args, kwargs)
        return outs

    def _transfer_async(
            self, src_array: NDArray, dst_dev: NDArray,
            src_stream=None, dst_stream=None) -> IncomingData:
        src_dev = src_array.device.id

        with Device(src_dev):
            if src_stream is None:
                src_stream = Stream()
            with src_stream:
                src_array = cupy.ascontiguousarray(src_array)
        with Device(dst_dev):
            if dst_stream is None:
                dst_stream = Stream()
            with dst_stream:
                dst_buf = cupy.full_like(src_array, -1)

        dtype, count = _get_nccl_dtype_and_count(src_array)
        nccl.groupStart()

        with Device(src_dev):
            self._comms[src_dev].send(
                src_array.data.ptr, count, dtype, dst_dev, src_stream.ptr)

        with Device(dst_dev):
            self._comms[dst_dev].recv(
                dst_buf.data.ptr, count, dtype, src_dev, dst_stream.ptr)

        nccl.groupEnd()
        return IncomingData(dst_stream, dst_buf)

    def _copy_to(
            self, src_array: NDArray, dst_array: NDArray) -> None:
        dst_dev = dst_array.device.id
        with Device(dst_dev):
            dst_array[:] = src_array.copy()

    def _apply_and_update_chunks(
            self, op_mode, shape, incomings,
            src_chunk, src_idx, dst_chunk, dst_idx):
        src_dev = src_chunk.device.id
        dst_dev = dst_chunk.device.id

        intersection = _index_intersection(src_idx, dst_idx, shape)
        if intersection is None:
            return
        src_new_idx = _index_for_subindex(src_idx, intersection, shape)
        dst_new_idx = _index_for_subindex(dst_idx, intersection, shape)

        with Device(src_dev):
            src_stream = Stream()

        incoming = self._transfer_async(
            src_chunk[src_new_idx], dst_dev, src_stream=src_stream)
        with Device(dst_dev):
            with incoming.stream:
                op_mode.func(incoming.data, dst_chunk[dst_new_idx],
                             incoming.data)

        incomings[dst_dev].append((incoming, dst_new_idx))

        if op_mode.identity is not None:
            with Device(src_dev):
                with src_stream:
                    src_chunk[src_new_idx] = op_mode.identity

    def _all_reduce_intersections(
            self, op_mode, shape, chunks, device_mapping, streams):
        chunks_list = list(chunks.items())
        incomings = {dev: [] for dev in device_mapping}

        # TODO: Parallelize
        for i in range(len(chunks_list)):
            src_dev, src_chunk = chunks_list[i]
            src_idx = device_mapping[src_dev]
            streams[src_dev].synchronize()

            for j in range(i + 1, len(chunks_list)):
                dst_dev, dst_chunk = chunks_list[j]
                dst_idx = device_mapping[dst_dev]

                with Device(src_dev):
                    for incoming, idx in incomings[src_dev]:
                        incoming.stream.synchronize()
                        src_chunk[idx] = incoming.data

                self._apply_and_update_chunks(
                    op_mode, shape, incomings,
                    src_chunk, src_idx, dst_chunk, dst_idx)

        # TODO: Parallelize
        for j in range(len(chunks_list)):
            src_dev, src_chunk = chunks_list[j]
            src_idx = device_mapping[src_dev]
            with Device(src_dev):
                for incoming, idx in incomings[src_dev]:
                    incoming.stream.synchronize()
                    src_chunk[idx] = incoming.data

            for i in range(j):
                dst_dev, dst_chunk = chunks_list[i]
                dst_idx = device_mapping[dst_dev]

                self._send_intersection(
                    shape, src_chunk, src_idx, dst_chunk, dst_idx)

    def _send_intersection(
            self, shape: tuple[int, ...],
            src_chunk: NDArray, src_idx: tuple[slice, ...],
            dst_chunk: NDArray, dst_idx: tuple[slice, ...]) -> None:
        """Write the intersection of chunks src_idx and dst_idx to the
        corresponding entries of dst_chunk. Note dst_chunk == self[dst_idx]."""

        # intersection == src_chunk[src_new_idx] == dst_chunk[dst_new_idx]
        intersection = _index_intersection(src_idx, dst_idx, shape)
        if intersection is None:
            return
        src_new_idx = _index_for_subindex(src_idx, intersection, shape)
        dst_new_idx = _index_for_subindex(dst_idx, intersection, shape)
        self._copy_to(src_chunk[src_new_idx], dst_chunk[dst_new_idx])

    def _send_intersection_async(
            self, shape: tuple[int, ...],
            src_chunk: NDArray, src_idx: tuple[slice, ...],
            dst_chunk: NDArray, dst_idx: tuple[slice, ...],
            src_stream: Optional[Stream] = None,
            dst_stream: Optional[Stream] = None,
        ) -> Optional[tuple[IncomingData, tuple[slice, ...]]]:
        """Write the intersection of chunks src_idx and dst_idx to the
        corresponding entries of dst_chunk. Note dst_chunk == self[dst_idx]."""

        # intersection == src_chunk[src_new_idx] == dst_chunk[dst_new_idx]
        intersection = _index_intersection(src_idx, dst_idx, shape)
        if intersection is None:
            return None

        dst_dev = dst_chunk.device.id
        src_new_idx = _index_for_subindex(src_idx, intersection, shape)
        dst_new_idx = _index_for_subindex(dst_idx, intersection, shape)

        return (
            self._transfer_async(src_chunk[src_new_idx], dst_dev,
                                 src_stream, dst_stream),
            dst_new_idx)

    def _set_identity_on_intersection(
            self, shape: tuple[int, ...], identity,
            a_chunk: NDArray, a_idx: tuple[slice, ...],
            b_idx: tuple[slice, ...]) -> None:
        intersection = _index_intersection(a_idx, b_idx, shape)
        if intersection is None:
            return
        a_new_idx = _index_for_subindex(a_idx, intersection, shape)
        with Device(a_chunk.device.id):
            a_chunk[a_new_idx] = identity

    def _set_identity_on_incoming(
            self, identity, dev,
            chunk: NDArray, incomings: list[tuple[IncomingData, Any]],
            stream: Optional[Stream] = None,
        ) -> Stream:
        if not stream:
            stream = Stream()
        for _, idx in incomings:
            with Device(dev):
                with stream:
                    chunk[idx] = identity
        return stream

    def __cupy_override_reduction_kernel__(
            self, kernel, axis, dtype, out, keepdims):
        # Assumption: kernel(x, y) == kernel(y, x) && kernel(x, x) == x e.g. max

        if out is not None:
            raise RuntimeError('Argument `out` is not supported')
        if keepdims:
            raise RuntimeError('`keepdims` is not supported')

        if kernel.name == 'cupy_max':
            mode = _max_mode
            identity = _min_value(self.dtype)
            chunks, chunk_streams = self._replica_mode_chunks()
        elif kernel.name == 'cupy_min':
            mode = _min_mode
            identity = _max_value(self.dtype)
            chunks, chunk_streams = self._replica_mode_chunks()
        # elif kernel.name == 'cupy_sum':
        #     mode = _sum_mode
        #     chunks = self._op_mode_chunks(mode)
        # elif kernel.name == 'cupy_prod':
        #     mode = _prod_mode
        #     chunks = self._op_mode_chunks(mode)
        else:
            raise RuntimeError(f'Unsupported kernel: {kernel.name}')

        for dev, stream in chunk_streams.items():
            self._set_identity_on_incoming(
                identity, dev, chunks[dev], self._incomings[dev], stream)

        shape = self.shape[:axis] + self.shape[axis+1:]
        new_dtype = None
        new_chunks = {}
        new_device_mapping = {}
        new_incomings = {dev: [] for dev in self._device_mapping}

        for dev, chunk in chunks.items():
            idx = self._device_mapping[dev]
            with Device(dev):
                with chunk_streams[dev]:
                    new_chunks[dev] = kernel(chunk, axis=axis, dtype=dtype)
            new_dtype = new_chunks[dev].dtype
            new_device_mapping[dev] = idx[:axis] + idx[axis+1:]

        for dev, incomings in self._incomings.items():
            with Device(dev):
                for incoming, idx in incomings:
                    stream = Stream()
                    stream.wait_event(incoming.stream.record())
                    with stream:
                        new_data = kernel(incoming.data, axis=axis, dtype=dtype)
                    new_idx = idx[:axis] + idx[axis+1:]
                    new_incomings[dev].append(
                        (IncomingData(stream, new_data), new_idx))

        return _DistributedArray(
            shape, new_dtype, new_chunks, new_device_mapping, mode, self._comms,
            new_incomings)

    def _copy_chunks(self) -> tuple[dict[int, NDArray], dict[int, Stream]]:
        chunks = {}
        streams = {}
        for dev, chunk in self._chunks.items():
            with Device(dev):
                streams[dev] = Stream()
                with streams[dev]:
                    chunks[dev] = chunk.copy()
        return chunks, streams

    def _replica_mode_chunks(self
                             ) -> tuple[dict[int, NDArray], dict[int, Stream]]:
        """Does not recessarily copy."""
        if self._mode is _replica_mode:
            return self._copy_chunks()

        self._wait_all_transfer()
        chunks, streams = self._copy_chunks()
        self._all_reduce_intersections(
            self._mode, self.shape, chunks, self._device_mapping, streams)
        return chunks, streams

    def _op_mode_chunks(self, op_mode) -> dict[int, NDArray]:
        if self._mode is op_mode:
            return self._chunks

        chunks = self._replica_mode_chunks()

        chunks_list = list(chunks.items())
        new_chunks = {}
        # TODO: Parallelize
        for i in range(len(chunks_list)):
            a_dev, a_chunk = chunks_list[i]
            with Device(a_dev):
                new_a_chunk = a_chunk.copy()
            a_idx = self._device_mapping[a_dev]

            for j in range(i + 1, len(chunks_list)):
                b_dev, _ = chunks_list[j]
                b_idx = self._device_mapping[b_dev]

                self._set_identity_on_intersection(
                    self.shape, op_mode.identity, new_a_chunk, a_idx, b_idx)

            new_chunks[a_dev] = new_a_chunk

        return new_chunks

    def to_replica_mode(self) -> '_DistributedArray':
        """Does not recessarily copy."""
        if self._mode is _replica_mode:
            return self
        else:
            chunks, streams = self._replica_mode_chunks()
            for s in streams.values():
                s.synchronize()
            return _DistributedArray(
                self.shape, self.dtype, chunks, self._device_mapping,
                _replica_mode, self._comms)

    def change_mode(self, mode) -> '_DistributedArray':
        if self._mode is mode:
            return self

        chunks = self._copy_chunks()
        self._all_reduce_intersections(
            self._mode, self.shape, chunks, self._device_mapping)
        return _DistributedArray(
            self.shape, self.dtype, chunks, self._device_mapping, _replica_mode,
            self._comms)

    def reshard(
        self, device_mapping: dict[int, Any]
    ) -> '_DistributedArray':
        # self._wait_all_transfer()

        # TODO: implement another version that reshards within the SUM mode
        arr = self.to_replica_mode()

        old_mapping = arr._device_mapping

        new_mapping = {dev: _convert_chunk_idx_to_slices(arr.shape, idx)
                       for dev, idx in device_mapping.items()}

        new_chunks = {}
        new_incoming = {dev: [] for dev in new_mapping}
        dst_streams = {}
        for dst_dev, dst_idx in new_mapping.items():
            with Device(dst_dev):
                dst_shape = _shape_after_indexing(arr.shape, dst_idx)
                dst_streams[dst_dev] = Stream()
                with dst_streams[dst_dev]:
                    new_chunks[dst_dev] = cupy.empty(
                        dst_shape, dtype=self.dtype)

        for src_dev, src_idx in old_mapping.items():
            src_array = arr._chunks[src_dev]
            src_collector = arr._collect_chunk(
                _replica_mode, src_array, arr._incomings[src_dev])
            with Device(src_dev):
                src_collected = src_collector.record()

            for dst_dev, dst_idx in new_mapping.items():
                dst_array = new_chunks[dst_dev]
                with Device(src_dev):
                    src_stream = Stream()
                    src_stream.wait_event(src_collected)

                incoming_and_idx = arr._send_intersection_async(
                    arr.shape, src_array, src_idx, dst_array, dst_idx,
                    src_stream=src_stream, dst_stream=dst_streams[dst_dev])
                if incoming_and_idx is not None:
                    new_incoming[dst_dev].append(incoming_and_idx)

        return _DistributedArray(
            arr.shape, arr.dtype, new_chunks, new_mapping, _replica_mode,
            self._comms, new_incoming)

    def asnumpy(self):
        self._wait_all_transfer()

        if self._mode is _replica_mode or self._mode.identity is None:
            np_array = numpy.empty(self.shape, dtype=self.dtype)
        else:
            np_array = numpy.full(self.shape, self.mode.identity, self.dtype)

        for dev, chunk_idx in self._device_mapping.items():
            if self._mode is _replica_mode or self._mode.identity is None:
                np_array[chunk_idx] = cupy.asnumpy(self._chunks[dev])
            else:
                self._mode.numpy_func(
                    np_array[chunk_idx], cupy.asnumpy(self._chunks[dev]),
                    out=np_array[chunk_idx])

        return np_array


def distributed_array(
        array: ArrayLike,
        device_mapping: dict[int, Any],
        mode: Mode = _replica_mode,
        comms = None):
    if not isinstance(array, (numpy.ndarray, cupy.ndarray)):
        array = numpy.array(array)
    elif mode is not _replica_mode:
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
            cp_chunks[dev] = cupy.array(chunk)
        if mode is not _replica_mode and mode.identity is not None:
            array[chunk_idx] = mode.identity
    return _DistributedArray(
        array.shape, array.dtype, cp_chunks, device_mapping, mode, comms)

