import dataclasses
import typing
from typing import Any, Callable, Final, Generic, Iterable, Optional, TypeVar
from typing_extensions import TypeGuard

from cupy.cuda import nccl
from cupy.cuda import Device, Event, Stream, get_current_stream

import cupy
import numpy

from numpy.typing import ArrayLike
from cupy_backends.cuda.api import runtime as cuda_runtime
from cupyx.distributed._nccl_comm import _nccl_dtypes
from cupyx.distributed import _linalg


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
    shape: tuple[int, ...],
) -> Optional[tuple[slice, ...]]:
    """Return None if empty."""
    ndim = len(shape)
    assert len(a_idx) == len(b_idx) == ndim
    result = tuple(_slice_intersection(a, b, length)
                   for a, b, length in zip(a_idx, b_idx, shape))

    def has_no_none(
            xs: tuple[Optional[slice], ...]) -> TypeGuard[tuple[slice, ...]]:
        return None not in xs

    if has_no_none(result):
        return result
    else:
        return None


def _index_for_subindex(
    a_idx: tuple[slice, ...], sub_idx: tuple[slice, ...],
    shape: tuple[int, ...],
) -> tuple[slice, ...]:
    ndim = len(shape)
    assert len(a_idx) == len(sub_idx) == ndim

    return tuple(_index_for_subslice(a, sub, length)
                   for a, sub, length in zip(a_idx, sub_idx, shape))


# Temporary helper function.
# Should be removed after implementing indexing
def _shape_after_indexing(
    outer_shape: tuple[int, ...],
    idx: tuple[slice, ...],
) -> tuple[int, ...]:
    shape = list(outer_shape)
    for i in range(len(idx)):
        start, stop, step = idx[i].indices(shape[i])
        shape[i] = (stop - start - 1) // step + 1
    return tuple(shape)


def _convert_chunk_idx_to_slices(
        shape: tuple[int, ...], idx: Any) -> tuple[slice, ...]:
    """Convert idx to type tuple[slice, ...] with length == ndim. Raise if empty
    or invalid. Negative slice steps are not allowed."""

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
            new_idx.append(idx[i])
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


_Scalar = TypeVar("_Scalar", bound=numpy.generic)


class _OpMode(Generic[_Scalar]):
    func: cupy.ufunc
    numpy_func: numpy.ufunc
    idempotent: bool
    identity_of: Callable[[numpy.dtype], _Scalar]

    def __init__(self, func_name: str, idempotent, identity_of):
        try:
            self.func = getattr(cupy, func_name)
            self.numpy_func = getattr(numpy, func_name)
        except AttributeError:
            raise RuntimeError('No such function exists')

        self.idempotent = idempotent
        self.identity_of = identity_of


_Mode = Optional[_OpMode[_Scalar]]


_REPLICA_MODE: Final[_Mode] = None


def _is_op_mode(mode: _Mode) -> TypeGuard[_OpMode]:
    return mode is not _REPLICA_MODE


_MODES: Final[dict[str, _Mode]] = {
    'replica': _REPLICA_MODE,
    'min': _OpMode('minimum', True, _max_value_of),
    'max': _OpMode('maximum', True, _min_value_of),
    'sum': _OpMode('add', False, _zero_value_of),
    'prod': _OpMode('multiply', False, _one_value_of),
}


@dataclasses.dataclass
class _ManagedData:
    """ND-array managed by a stream."""
    data: cupy.ndarray
    ready: Event
    prevent_gc: Any

    def __init__(
            self, data: cupy.ndarray, ready: Event, prevent_gc: Any = None,
        ) -> None:
        self.data = data
        self.ready = ready
        self.prevent_gc = prevent_gc

    def copy(self) -> '_ManagedData':
        with self.data.device:
            stream = get_current_stream()
            new_data = self.data.copy()
            self.ready.record(stream)
            return _ManagedData(new_data, stream.record(), self.prevent_gc)


@dataclasses.dataclass
class _DataTransfer:
    """ND-array managed by a stream."""
    data: cupy.ndarray
    ready: Event
    prevent_gc: Any = None


# Overwrite in replica mode, apply in op mode
_PartialUpdate = tuple[_DataTransfer, tuple[slice, ...]]


@dataclasses.dataclass
class _Chunk(_ManagedData):
    index: tuple[slice, ...]
    updates: list[_PartialUpdate]

    def __init__(
            self, data: cupy.ndarray, ready: Event, index: tuple[slice, ...],
            updates: Optional[list[_PartialUpdate]] = None,
            prevent_gc: Any = None) -> None:
        super().__init__(data, ready, prevent_gc)
        self.index = index
        self.updates = updates if updates is not None else []

    def copy(self) -> '_Chunk':
        managed_data = super().copy()

        return _Chunk(managed_data.data, managed_data.ready,
                      self.index, list(self.updates),
                      prevent_gc=self.prevent_gc)

    def apply_updates(self, mode: _Mode) -> None:
        """Apply all updates in-place."""
        if len(self.updates) == 0:
            return

        with self.data.device:
            stream = cupy.cuda.get_current_stream()
            stream.wait_event(self.ready)
            for new_data, idx in self.updates:
                stream.wait_event(new_data.ready)
                with stream:
                    if _is_op_mode(mode):
                        mode.func(
                            self.data[idx], new_data.data, self.data[idx])
                    else:
                        self.data[idx] = new_data.data

            self.ready.record(stream)
            prevent_gc = self.updates
            prevent_gc.append(self.prevent_gc)
            self.prevent_gc = prevent_gc
            self.updates = []


def _create_communicators(
        devices: Iterable[int]
    ) -> dict[int, nccl.NcclCommunicator]:  #type: ignore
    comms_list = nccl.NcclCommunicator.initAll(list(devices))
    return {comm.device_id(): comm for comm in comms_list}


def _transfer_async(
        src_comm: nccl.NcclCommunicator,
        src_stream: Stream,
        src_data: _ManagedData,
        dst_comm: nccl.NcclCommunicator,
        dst_stream: Stream,
        dst_dev: int) -> _DataTransfer:
    src_dev = src_data.data.device.id

    if src_dev == dst_dev:
        with Device(src_dev):
            return _DataTransfer(
                src_data.data, src_data.ready, src_data.prevent_gc)

    with Device(src_dev):
        src_stream.wait_event(src_data.ready)
        with src_stream:
            src_array = cupy.ascontiguousarray(src_data.data)
    with Device(dst_dev):
        with dst_stream:
            dst_buf = cupy.empty(src_array.shape, src_array.dtype)

    dtype, count = _get_nccl_dtype_and_count(src_array)
    nccl.groupStart()   # type: ignore

    with Device(src_dev):
        src_comm.send(src_array.data.ptr, count, dtype,
                      dst_comm.rank_id(), src_stream.ptr)
        src_stream.record(src_data.ready)

    with Device(dst_dev):
        dst_comm.recv(dst_buf.data.ptr, count, dtype,
                      src_comm.rank_id(), dst_stream.ptr)

        nccl.groupEnd()     # type: ignore
        return _DataTransfer(dst_buf, dst_stream.record(),
                             prevent_gc=(src_data, src_array))


class _DistributedArray(cupy.ndarray, Generic[_Scalar]):
    _chunks_map: dict[int, list[_Chunk]]
    _streams: dict[int, Stream]
    _mode: _Mode
    _comms: dict[int, nccl.NcclCommunicator]    # type: ignore
    _mem: cupy.cuda.Memory

    def __new__(
        cls, shape: tuple[int, ...], dtype: Any,
        chunks_map: dict[int, list[_Chunk]],
        mode: _Mode = _REPLICA_MODE,
        comms: Optional[dict[int, nccl.NcclCommunicator]]   # type: ignore
            = None,
    ) -> '_DistributedArray':
        mem = _MultiDeviceDummyMemory(0)
        memptr = _MultiDeviceDummyPointer(mem, 0)
        obj = super().__new__(cls, shape, dtype, memptr=memptr)
        obj._chunks_map = chunks_map
        obj._streams = {}
        for dev in chunks_map.keys():
            with Device(dev):
                obj._streams[dev] = Stream()
        obj._mode = mode
        if comms:
            obj._comms = comms
        elif nccl.available:
            comms_list = nccl.NcclCommunicator.initAll(     # type: ignore
                list(chunks_map.keys()))
            obj._comms = {dev: comm
                          for dev, comm in zip(chunks_map.keys(), comms_list)}
        else:
            # TODO: support environments where NCCL is unavailable
            raise RuntimeError('NCCL is unavailable')
        return obj

    def __array_finalize__(self, obj):
        # TODO set sensible defualts
        if obj is None:
            return
        self._chunks_map = getattr(obj, '_chunks_map', None)
        self._streams = getattr(obj, '_streams', None)
        self._mode = getattr(obj, '_mode', None)
        self._comms = getattr(obj, '_comms', None)
        self._mem = getattr(obj, '_mem', None)

    @property
    def mode(self) -> str:
        assert self._mode in _MODES.values()
        for mode_str, mode_obj in _MODES.items():
            if self._mode is mode_obj:
                return mode_str

    @property
    def devices(self) -> Iterable[int]:
        return self._chunks_map.keys()

    @property
    def index_map(self) -> dict[int, list[tuple[slice, ...]]]:
        return {dev: [chunk.index for chunk in chunks]
                for dev, chunks in self._chunks_map.items()}

    def _prepare_comms(self, devices: Iterable[int]) -> None:
        devices = self._chunks_map.keys() | devices
        if devices == self._chunks_map.keys():
            return
        self._comms = _create_communicators(devices)

    def _count_chunks_on_devices(self, dist_args) -> dict[int, int]:
        counts = {}
        for _, arg in dist_args:
            for dev, chunks in arg._chunks_map.items():
                if dev not in counts:
                    counts[dev] = len(chunks)
                if counts[dev] != len(chunks):
                    raise RuntimeError('Chunks have different shapes')
        return counts

    def _get_chunk_data(self, dev, i=0) -> _ManagedData:
        return self._chunks_map[dev][i].data

    def _stream_for(self, dev: int) -> Stream:
        if dev not in self._streams:
            with Device(dev):
                self._streams[dev] = Stream()

        return self._streams[dev]

    def _copy_chunk(self, chunk: _Chunk) -> _Chunk:
        with chunk.data.device:
            return chunk.copy()

    def _apply_updates(self, chunk: _Chunk, mode: _Mode) -> None:
        chunk.apply_updates(mode)

    def wait_all_transfer(self) -> None:
        """Block until all inner update_map are done."""
        transfer_events = []

        for chunks in self._chunks_map.values():
            for chunk in chunks:
                self._apply_updates(chunk, self._mode)
            transfer_events.append(chunk.ready)

        for e in transfer_events:
            e.synchronize()

    def _prepare_args(
        self, dist_args: list[tuple[int | str, '_DistributedArray']],
        regular_args: list[tuple[int | str, cupy.ndarray]],
        dev: int, chunk_i: int,
    ) -> list[tuple[int | str, _ManagedData]]:
        # Dist arrays must have chunk_map of compatible shapes, otherwise
        # hard error.
        # In case that they are of different, but broadcastable shapes
        # Data movement may be needed
        # Currently: Support only same shape chunk_map
        args = []
        for i, arg in dist_args:
            chunk = arg._chunks_map[dev][chunk_i]
            args.append(
                (i, _ManagedData(chunk.data, chunk.ready, chunk.prevent_gc)))

        # Case of X.T and other data movement requiring cases not supported
        # TODO(ecastill) add support for operands being non distributed arrays
        # 1. Check if the regular arrays are on the specified device or
        #    peer access is enabled
        # 2. Check that their shape is compatible with the chunk_map
        #    distributed arrays
        # 3. Create views of this array and copy to the given device if needed
        #    so that the chunk_map in the distributed operate with the right slice
        if len(regular_args) > 0:
            raise RuntimeError(
                'Mix `cupy.ndarray` with distributed arrays is currently not'
                ' supported')

        return args

    def _prepare_updates(
        self, dist_args: list[tuple[int | str, '_DistributedArray']],
        dev: int, chunk_i: int,
    ) -> tuple[Optional[int | str], list[_PartialUpdate]]:
        index = None
        updates: list[_PartialUpdate] = []
        at_most_one_update = True
        for i, arg in dist_args:
            updates_now = arg._chunks_map[dev][chunk_i].updates
            if updates_now:
                if updates:
                    at_most_one_update = False
                    break
                index = i
                updates = updates_now

        # If there is at most one array with partial updates, we return them
        # and apply the element-wise kernel without actually propagating
        # those updates. Otherwise we propagate them beforehand.
        # TODO check if it really gives any speedup
        if at_most_one_update:
            return index, updates

        for i, arg in dist_args:
            for chunk in arg._chunks_map[dev]:
                self._apply_updates(chunk, _REPLICA_MODE)
        return None, []

    def _execute_kernel(
        self, kernel, args: tuple[Any, ...], kwargs: dict[str, Any],
    ) -> '_DistributedArray':
        distributed_arrays: list[tuple[int | str, '_DistributedArray']] = []
        regular_arrays: list[tuple[int | str, cupy.ndarray]] = []
        i: int | str
        index_map = self.index_map
        for i, arg in enumerate(args):
            if arg.shape != self.shape:
                # TODO support broadcasting
                raise RuntimeError('Mismatched shapes')

            if isinstance(arg, _DistributedArray):
                if arg.index_map != index_map:
                    # TODO enable p2p access
                    raise RuntimeError('Mismatched index_map')

                distributed_arrays.append((i, arg.to_replica_mode()))
            elif isinstance(arg, cupy.ndarray):
                regular_arrays.append((i, arg))

        # Do it for kwargs too
        for k, arg in kwargs.items():
            if arg.shape != self.shape:
                # TODO support broadcasting
                raise RuntimeError('Mismatched shapes')
            if arg.index_map != index_map:
                # TODO enable p2p access
                raise RuntimeError('Mismatched index_map')
            if isinstance(arg, _DistributedArray):
                distributed_arrays.append((k, arg))
            elif isinstance(arg, cupy.ndarray):
                regular_arrays.append((k, arg))

        args = list(args)
        chunk_counts = self._count_chunks_on_devices(distributed_arrays)
        new_dtype = None
        new_chunks_map: dict[int, list[_Chunk]] = {}
        for dev, chunk_count in chunk_counts.items():
            new_chunks_map[dev] = []
            with Device(dev):
                stream = get_current_stream()

                for chunk_i in range(chunk_count):
                    array_args = self._prepare_args(
                        distributed_arrays, regular_arrays, dev, chunk_i)

                    incoming_index, update_map = self._prepare_updates(
                        distributed_arrays, dev, chunk_i)

                    for i, arg in array_args:
                        stream.wait_event(arg.ready)
                        if isinstance(i, int):
                            args[i] = arg.data
                        else:
                            kwargs[i] = arg.data

                    new_data = kernel(*args, **kwargs)

                    new_dtype = new_data.dtype
                    new_chunk = _Chunk(
                        new_data, stream.record(),
                        index_map[dev][chunk_i],
                        prevent_gc=(args, kwargs))
                    new_chunks_map[dev].append(new_chunk)

                    if len(update_map) == 0:
                        continue

                    incoming_index = typing.cast(int | str, incoming_index)

                    args_slice = [None] * len(args)
                    kwargs_slice: dict[str, cupy.ndarray] = {}
                    for update, idx in update_map:
                        for i, arg in enumerate(args):
                            args_slice[i] = arg[idx]
                        for k, arg in kwargs.items():
                            kwargs_slice[k] = arg[idx]

                        if isinstance(incoming_index, int):
                            args_slice[incoming_index] = update.data
                        else:
                            kwargs_slice[incoming_index] = update.data

                        stream.wait_event(update.ready)
                        new_data = kernel(*args_slice, **kwargs_slice)
                        execution_done = stream.record()

                        data_transfer = _DataTransfer(
                            new_data, execution_done,
                            prevent_gc=(args_slice, kwargs_slice))
                        new_chunk.updates.append((data_transfer, idx))

        for chunks in new_chunks_map.values():
            for chunk in chunks:
                if not isinstance(chunk.data, cupy.ndarray):
                    raise RuntimeError(
                        'Kernels returning other than signle array are not'
                        ' supported')

        return _DistributedArray(
            self.shape, new_dtype, new_chunks_map, _REPLICA_MODE, self._comms)

    def __cupy_override_elementwise_kernel__(self, kernel, *args, **kwargs):
        # This defines a protocol to be called from elementwise kernel
        # to override some of the ops done there
        outs = self._execute_kernel(kernel, args, kwargs)
        return outs

    def _transfer_async(
            self, src_data: _ManagedData, dst_dev: int) -> _DataTransfer:
        src_dev = src_data.data.device.id
        return _transfer_async(
            self._comms[src_dev], self._stream_for(src_dev), src_data,
            self._comms[dst_dev], self._stream_for(dst_dev), dst_dev)

    def _apply_and_update_chunks(
        self, op_mode: _OpMode, shape: tuple[int, ...],
        src_chunk: _Chunk, dst_chunk: _Chunk,
    ) -> None:
        """Apply `src_chunk` onto `dst_chunk` in `op_mode`.
        There must not be any undone partial update to src_chunk."""
        src_dev = src_chunk.data.device.id
        dst_dev = dst_chunk.data.device.id
        src_idx = src_chunk.index
        dst_idx = dst_chunk.index

        intersection = _index_intersection(src_idx, dst_idx, shape)
        if intersection is None:
            return
        src_new_idx = _index_for_subindex(src_idx, intersection, shape)
        dst_new_idx = _index_for_subindex(dst_idx, intersection, shape)

        data_to_transfer = _ManagedData(
            src_chunk.data[src_new_idx], src_chunk.ready, src_chunk.prevent_gc)

        if not op_mode.idempotent:
            with cupy.cuda.Device(src_dev):
                data_to_transfer = data_to_transfer.copy()
            copy_done = data_to_transfer.ready

        update = self._transfer_async(data_to_transfer, dst_dev)
        dst_chunk.updates.append((update, dst_new_idx))

        if not op_mode.idempotent:
            dtype = src_chunk.data.dtype
            with Device(src_dev):
                stream = get_current_stream()
                stream.wait_event(copy_done)
                src_chunk.data[src_new_idx] = op_mode.identity_of(dtype)
                stream.record(src_chunk.ready)

    def _all_reduce_intersections(
        self, op_mode: _OpMode, shape: tuple[int, ...],
        chunk_map: dict[int, list[_Chunk]],
    ) -> None:
        chunks_list = [chunk for chunks in chunk_map.values()
                             for chunk in chunks]

        # TODO flatten this loop somehow
        for i in range(len(chunks_list)):
            src_chunk = chunks_list[i]
            self._apply_updates(src_chunk, op_mode)

            for j in range(i + 1, len(chunks_list)):
                dst_chunk = chunks_list[j]
                self._apply_and_update_chunks(
                    op_mode, shape, src_chunk, dst_chunk)

        for j in range(len(chunks_list) - 1, -1, -1):
            src_chunk = chunks_list[j]
            self._apply_updates(src_chunk, _REPLICA_MODE)

            for i in range(j):
                dst_chunk = chunks_list[i]
                self._copy_on_intersection(shape, src_chunk, dst_chunk)

    def _copy_on_intersection(
        self, shape: tuple[int, ...],
        src_chunk: _Chunk, dst_chunk: _Chunk,
    ) -> None:
        assert len(src_chunk.updates) == 0

        src_idx = src_chunk.index
        dst_idx = dst_chunk.index
        intersection = _index_intersection(src_idx, dst_idx, shape)
        if intersection is None:
            return

        dst_dev = dst_chunk.data.device.id
        src_new_idx = _index_for_subindex(src_idx, intersection, shape)
        dst_new_idx = _index_for_subindex(dst_idx, intersection, shape)

        src_partial_chunk = _ManagedData(
            src_chunk.data[src_new_idx], src_chunk.ready, src_chunk.prevent_gc)
        update = self._transfer_async(src_partial_chunk, dst_dev)
        dst_chunk.updates.append((update, dst_new_idx))

    def _set_identity_on_intersection(
        self, shape: tuple[int, ...], identity: _Scalar,
        a_chunk: _Chunk, b_idx: tuple[slice, ...],
    ) -> None:
        a_idx = a_chunk.index
        intersection = _index_intersection(a_idx, b_idx, shape)
        if intersection is None:
            return
        a_new_idx = _index_for_subindex(a_idx, intersection, shape)
        with a_chunk.data.device:
            stream = get_current_stream()
            stream.wait_event(a_chunk.ready)
            a_chunk.data[a_new_idx] = identity
            stream.record(a_chunk.ready)

    def _set_identity_on_ignored_entries(
            self, identity: _Scalar, chunk: _Chunk) -> None:
        with chunk.data.device:
            stream = get_current_stream()
            stream.wait_event(chunk.ready)
            for _, idx in chunk.updates:
                chunk.data[idx] = identity
            stream.record(chunk.ready)

    def __cupy_override_reduction_kernel__(
            self, kernel, axis, dtype, out, keepdims) -> Any:
        # This defines a protocol to be called from reduction functions
        # to override some of the ops done there
        if out is not None:
            raise RuntimeError('Argument `out` is not supported')
        if keepdims:
            raise RuntimeError('`keepdims` is not supported')

        overwrites = False
        if kernel.name == 'cupy_max':
            mode = _MODES['max']
            if self._mode is mode:
                chunks_map = self._chunks_map
            else:
                chunks_map = self._replica_mode_chunks_map()
                overwrites = True
        elif kernel.name == 'cupy_min':
            mode = _MODES['min']
            if self._mode is mode:
                chunks_map = self._chunks_map
            else:
                chunks_map = self._replica_mode_chunks_map()
                overwrites = True
        elif kernel.name == 'cupy_sum':
            mode = typing.cast(_OpMode, _MODES['sum'])
            chunks_map = self._op_mode_chunks_map(mode)
        elif kernel.name == 'cupy_prod':
            mode = typing.cast(_OpMode, _MODES['prod'])
            chunks_map = self._op_mode_chunks_map(mode)
        else:
            raise RuntimeError(f'Unsupported kernel: {kernel.name}')

        if overwrites:
            mode = typing.cast(_OpMode, mode)
            identity = mode.identity_of(self.dtype)
            for chunks in chunks_map.values():
                for i in range(len(chunks)):
                    if len(chunks[i].updates) == 0:
                        continue
                    chunks[i] = self._copy_chunk(chunks[i])
                    self._set_identity_on_ignored_entries(identity, chunks[i])

        shape = self.shape[:axis] + self.shape[axis+1:]
        new_dtype = None
        new_chunks_map: dict[int, list[_Chunk]] = {}

        for dev, chunks in chunks_map.items():
            new_chunks_map[dev] = []
            for chunk in chunks:
                with Device(dev):
                    execution_stream = get_current_stream()
                    execution_stream.wait_event(chunk.ready)
                    new_data = cupy.atleast_1d(
                        kernel(chunk.data, axis=axis, dtype=dtype))

                    new_dtype = new_data.dtype
                    new_index = chunk.index[:axis] + chunk.index[axis+1:]
                    new_updates: list[_PartialUpdate] = []
                    new_chunks_map[dev].append(
                        _Chunk(new_data, execution_stream.record(),
                               new_index, new_updates,
                               prevent_gc=chunk.prevent_gc))

                    if len(chunk.updates) == 0:
                        continue

                    for update, update_index in chunk.updates:
                        execution_stream.wait_event(update.ready)
                        new_update_data = cupy.atleast_1d(
                            kernel(update.data, axis=axis, dtype=dtype))

                        data_transfer = _DataTransfer(
                            new_update_data, execution_stream.record(),
                            prevent_gc=update.prevent_gc)
                        new_index = update_index[:axis] + update_index[axis+1:]
                        new_updates.append((data_transfer, new_index))

        return _DistributedArray(
            shape, new_dtype, new_chunks_map, mode, self._comms)

    def _copy_chunks_map(self) -> dict[int, list[_Chunk]]:
        return {dev: [self._copy_chunk(chunk) for chunk in chunks]
                for dev, chunks in self._chunks_map.items()}

    def _copy_chunks_map_replica_mode(self) -> dict[int, list[_Chunk]]:
        """Return a copy of the chunks_map in the replica mode."""
        chunks_map = self._copy_chunks_map()
        if _is_op_mode(self._mode):
            self._all_reduce_intersections(
                self._mode, self.shape, chunks_map)
        return chunks_map

    def _replica_mode_chunks_map(self) -> dict[int, list[_Chunk]]:
        """Return a view or a copy of the chunks_map in the replica mode."""
        if self._mode is _REPLICA_MODE:
            return self._chunks_map

        if len(self._chunks_map) == 1:
            chunks, = self._chunks_map.values()
            if len(chunks) == 1:
                self._apply_updates(chunks[0], self._mode)
                return self._chunks_map

        return self._copy_chunks_map_replica_mode()

    def _op_mode_chunks_map(self, op_mode: _OpMode) -> dict[int, list[_Chunk]]:
        """Return a view or a copy of the chunks_map in the given mode."""
        if self._mode is op_mode:
            return self._chunks_map

        if len(self._chunks_map) == 1:
            chunks, = self._chunks_map.values()
            if len(chunks) == 1:
                self._apply_updates(chunks[0], self._mode)
                return self._chunks_map

        chunks_map = self._copy_chunks_map_replica_mode()

        for chunks in chunks_map.values():
            for chunk in chunks:
                self._apply_updates(chunk, _REPLICA_MODE)

        chunks_list = [chunk for chunks in chunks_map.values()
                             for chunk in chunks]
        identity = op_mode.identity_of(self.dtype)

        # TODO: Parallelize
        for i in range(len(chunks_list)):
            a_chunk = chunks_list[i]
            for j in range(i + 1, len(chunks_list)):
                b_chunk = chunks_list[j]
                self._set_identity_on_intersection(
                    self.shape, identity, a_chunk, b_chunk.index)

        return chunks_map

    def to_replica_mode(self) -> '_DistributedArray':
        """Return a view or a copy of self in the replica mode."""
        if self._mode is _REPLICA_MODE:
            return self
        else:
            chunks_map = self._replica_mode_chunks_map()
            return _DistributedArray(
                self.shape, self.dtype, chunks_map, _REPLICA_MODE, self._comms)

    def change_mode(self, mode: str) -> '_DistributedArray':
        """Return a view or a copy of self in the given mode."""
        if mode not in _MODES:
            raise RuntimeError(f'`mode` must be one of {list(_MODES)}')

        mode_obj = _MODES[mode]
        if mode_obj is self._mode:
            return self

        if _is_op_mode(mode_obj):
            chunks_map = self._op_mode_chunks_map(mode_obj)
        else:
            chunks_map = self._replica_mode_chunks_map()
        return _DistributedArray(
            self.shape, self.dtype, chunks_map, mode_obj, self._comms)

    def reshard(
            self, index_map: dict[int, Any]) -> '_DistributedArray':
        """Return a view or a copy of self with the given index_map."""
        self._prepare_comms(index_map.keys())

        new_index_map: dict[int, list[tuple[slice, ...]]] = {}
        for dev, idxs in index_map.items():
            if not isinstance(idxs, list):
                idxs = [idxs]
            for i in range(len(idxs)):
                idxs[i] = _convert_chunk_idx_to_slices(self.shape, idxs[i])
            idxs.sort(key=lambda slices:
                      [s.indices(l) for s, l in zip(slices, self.shape)])
            new_index_map[dev] = idxs

        if new_index_map == self.index_map:
            return self

        old_chunks_map = self._chunks_map
        new_chunks_map: dict[int, list[_Chunk]] = {}

        if _is_op_mode(self._mode):
            identity: _Scalar = self._mode.identity_of(self.dtype)

        for dev, idxs in new_index_map.items():
            new_chunks_map[dev] = []

            for idx in idxs:
                with Device(dev):
                    dst_shape = _shape_after_indexing(self.shape, idx)
                    stream = get_current_stream()
                    if self._mode is _REPLICA_MODE:
                        data = cupy.empty(dst_shape, self.dtype)
                    else:
                        data = cupy.full(dst_shape, identity, self.dtype)
                    data = cupy.atleast_1d(data)
                    new_chunk = _Chunk(data, stream.record(), idx)
                    new_chunks_map[dev].append(new_chunk)

        for src_chunks in old_chunks_map.values():
            for src_chunk in src_chunks:
                self._apply_updates(src_chunk, self._mode)

                if _is_op_mode(self._mode):
                    src_chunk = self._copy_chunk(src_chunk)

                for dst_chunks in new_chunks_map.values():
                    for dst_chunk in dst_chunks:
                        if _is_op_mode(self._mode):
                            self._apply_and_update_chunks(
                                self._mode, self.shape, src_chunk, dst_chunk)
                        else:
                            self._copy_on_intersection(
                                self.shape, src_chunk, dst_chunk)

        return _DistributedArray(
            self.shape, self.dtype, new_chunks_map, self._mode, self._comms)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc.__name__ == 'matmul' and method == '__call__':
            return _linalg.matmul(*inputs, **kwargs)
        return NotImplemented

    def __matmul__(x, y):
        if isinstance(y, _DistributedArray):
            return _linalg.matmul(x, y)
        else:
            return NotImplemented

    def _map_chunks(
            self,
            f: Callable[[_Chunk], _Chunk],
            chunks_map: dict[int, list[_Chunk]],
            ) -> dict[int, list[_Chunk]]:
        new_chunks_map = {}
        for dev, chunks in chunks_map.items():
            new_chunks_map[dev] = [f(chunk) for chunk in chunks]
        return new_chunks_map

    def _change_shape(
        self,
        f_shape: Callable[[tuple[int, ...]], tuple[int, ...]],
        f_idx: Callable[[tuple[slice, ...]], tuple[slice, ...]],
    ) -> '_DistributedArray':
        def apply_to_chunk(chunk: _Chunk) -> _Chunk:
            data = chunk.data.reshape(f_shape(chunk.data.shape))
            index = f_idx(chunk.index)
            updates = [(data, f_idx(idx))
                       for data, idx in chunk.updates]
            return _Chunk(data, chunk.ready, index, updates, chunk.prevent_gc)

        chunks_map = self._map_chunks(apply_to_chunk, self._chunks_map)
        return _DistributedArray(
            f_shape(self.shape), self.dtype, chunks_map,
            self._mode, self._comms)

    def _prepend_one_to_shape(self) -> '_DistributedArray':
        """Return a view with (1,) prepended to its shape."""
        return self._change_shape(
            lambda shape: (1,) + shape,
            lambda idx: (slice(None),) + idx)

    def _append_one_to_shape(self) -> '_DistributedArray':
        """Return a view with (1,) apppended to its shape."""
        return self._change_shape(
            lambda shape: shape + (1,),
            lambda idx: idx + (slice(None),))

    def _pop_from_shape(self) -> '_DistributedArray':
        """Return a view with the last element of shape popped. The last element
        of shape must be equal to 1."""
        assert self.shape[-1] == 1

        return self._change_shape(
            lambda shape: shape[:-1],
            lambda idx: idx[:-1])

    def _pop_front_from_shape(self) -> '_DistributedArray':
        """Return a view with the first element of shape popped. The first
        element of shape must be equal to 1."""
        assert self.shape[0] == 1

        return self._change_shape(
            lambda shape: shape[1:],
            lambda idx: idx[1:])

    def asnumpy(self) -> numpy.ndarray:
        for chunks in self._chunks_map.values():
            for chunk in chunks:
                self._apply_updates(chunk, self._mode)

        if _is_op_mode(self._mode):
            identity = self._mode.identity_of(self.dtype)
            np_array = numpy.full(self.shape, identity, self.dtype)
        else:
            np_array = numpy.empty(self.shape, dtype=self.dtype)

        # We expect np_array[idx] to return a view, but this is not true if
        # np_array is 0-dimensional.
        np_array = numpy.atleast_1d(np_array)

        for chunks in self._chunks_map.values():
            for chunk in chunks:
                chunk.ready.synchronize()
                idx = chunk.index
                if _is_op_mode(self._mode):
                    self._mode.numpy_func(
                        np_array[idx], cupy.asnumpy(chunk.data), np_array[idx])
                else:
                    np_array[idx] = cupy.asnumpy(chunk.data)

        # Undo numpy.atleast_1d
        return np_array.reshape(self.shape)


def distributed_array(
    array: ArrayLike,
    index_map: dict[int, Any],
    mode: str = 'replica',
    devices: Optional[int | Iterable[int]] = None,
    comms: Optional[dict[int, nccl.NcclCommunicator]] = None, # type: ignore
) -> _DistributedArray:
    if devices is not None and comms is not None:
        raise RuntimeError('Only one of devices and comms can be specified')

    if comms is None:
        # Initialize devices: Iterable[int]
        if devices is None:
            devices = index_map.keys()
            if isinstance(array, cupy.ndarray):
                devices |= {array.device.id}
        elif isinstance(devices, int):
            devices = range(devices)

        comms = _create_communicators(devices)

        if (isinstance(array, cupy.ndarray)
                and array.device.id not in comms.keys()):
            raise RuntimeError(
                'No communicator for transfer from the given array')

    if isinstance(array, _DistributedArray):
        if array.mode != mode:
            array = array.change_mode(mode)
        if array.index_map != index_map:
            array = array.reshard(index_map)
        return _DistributedArray(
            array.shape, array.dtype, array._chunks_map, array.mode, comms)

    if not isinstance(array, (numpy.ndarray, cupy.ndarray)):
        array = numpy.array(array)
    elif mode != 'replica':
        array = array.copy()

    new_index_map: dict[int, list[tuple[slice, ...]]] = {}
    for dev, idxs in index_map.items():
        if not isinstance(idxs, list):
            idxs = [idxs]
        for i in range(len(idxs)):
            idxs[i] = _convert_chunk_idx_to_slices(array.shape, idxs[i])
        idxs.sort(key=lambda slices:
                    [s.indices(l) for s, l in zip(slices, array.shape)])
        new_index_map[dev] = idxs

    if isinstance(array, cupy.ndarray):
        src_dev = array.device.id
        src_stream = get_current_stream()

        def make_chunk(dst_dev, idx, src_array):
            with src_array.device:
                src_array = cupy.ascontiguousarray(src_array)
                src_data = _ManagedData(src_array, src_stream.record(),
                                        prevent_gc=src_array)
            with Device(dst_dev):
                dst_stream = get_current_stream()
                copied = _transfer_async(
                    comms[src_dev], src_stream, src_data,
                    comms[dst_dev], dst_stream, dst_dev)
                return _Chunk(copied.data, copied.ready, idx,
                              prevent_gc=src_data)

    else:
        def make_chunk(dev, idx, array):
            with Device(dev):
                stream = get_current_stream()
                copied = cupy.array(array)
                return _Chunk(copied, stream.record(), idx,
                              prevent_gc=array)

    mode_obj = _MODES[mode]
    chunks_map: dict[int, list[_Chunk]] = {}
    for dev, idxs in new_index_map.items():
        chunks_map[dev] = []

        for i, idx in enumerate(idxs):
            chunk_data = array[idx]
            chunk = make_chunk(dev, idx, chunk_data)
            chunks_map[dev].append(chunk)
            if mode_obj is not None and not mode_obj.idempotent:
                array[idx] = mode_obj.identity_of(array.dtype)

    return _DistributedArray(
        array.shape, array.dtype, chunks_map, mode_obj, comms)
