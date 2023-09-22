import dataclasses
import typing
from typing import Any, Callable, Final, Iterable, Optional, TypeVar, Union
from typing_extensions import TypeGuard

from cupy.cuda import nccl
if nccl.available:
    from cupy.cuda.nccl import NcclCommunicator     # type: ignore
else:
    class NcclCommunicator:
        pass

from cupy.cuda import Device, Event, Stream, get_current_stream

import numpy
import cupy
from cupy import _core

from numpy.typing import ArrayLike

from cupyx.distributed._nccl_comm import _get_nccl_dtype_and_count
from cupyx.distributed import _linalg
from cupyx.distributed import _index_arith


class _MultiDeviceDummyMemory(cupy.cuda.Memory):
    pass


class _MultiDeviceDummyPointer(cupy.cuda.MemoryPointer):
    @property
    def device(self) -> Device:
        # This override is needed to assign an invalid device id
        # Since the array is not residing in a single device now
        return Device(-1)


def _min_value_of(dtype):
    if dtype.kind in 'biu':
        return dtype.type(cupy.iinfo(dtype).min)
    elif dtype.kind in 'f':
        return dtype.type(-cupy.inf)
    else:
        raise RuntimeError(f'Unsupported type: {dtype}')


def _max_value_of(dtype):
    if dtype.kind in 'biu':
        return dtype.type(cupy.iinfo(dtype).max)
    elif dtype.kind in 'f':
        return dtype.type(cupy.inf)
    else:
        raise RuntimeError(f'Unsupported type: {dtype}')


def _zero_value_of(dtype):
    return dtype.type(0)


def _one_value_of(dtype):
    return dtype.type(1)


class _OpMode:
    func: cupy.ufunc
    numpy_func: numpy.ufunc
    idempotent: bool
    identity_of: Callable

    _T = TypeVar('_T', bound=numpy.generic)
    def __init__(
        self, func_name: str, idempotent: bool, identity_of: Callable,
    ) -> None:
        try:
            self.func = getattr(cupy, func_name)
            self.numpy_func = getattr(numpy, func_name)
        except AttributeError:
            raise RuntimeError('No such function exists')

        self.idempotent = idempotent
        self.identity_of = identity_of


_Mode = Optional[_OpMode]


_REPLICA_MODE: Final[_Mode] = None


def _is_op_mode(mode: _Mode) -> TypeGuard[_OpMode]:
    return mode is not _REPLICA_MODE


_MODES: Final[dict[str, _Mode]] = {
    'replica': _REPLICA_MODE,
    'min':  _OpMode('minimum',  True,  _max_value_of),
    'max':  _OpMode('maximum',  True,  _min_value_of),
    'sum':  _OpMode('add',      False, _zero_value_of),
    'prod': _OpMode('multiply', False, _one_value_of),
}


@dataclasses.dataclass
class _ManagedData:
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
            stream.wait_event(self.ready)
            new_data = self.data.copy()
            self.ready.record(stream)
            return _ManagedData(new_data, stream.record(), self.prevent_gc)


@dataclasses.dataclass
class _DataTransfer:
    data: cupy.ndarray
    ready: Event
    prevent_gc: Any = None


# Overwrite in replica mode, apply in op mode
_PartialUpdate = tuple[_DataTransfer, tuple[slice, ...]]


@dataclasses.dataclass
class _DataPlaceholder:
    shape: tuple[int, ...]
    device: Device

    def copy(self) -> '_DataPlaceholder':
        return self

    def reshape(self, new_shape: tuple[int, ...]) -> '_DataPlaceholder':
        return _DataPlaceholder(new_shape, self.device)


@dataclasses.dataclass
class _Chunk:
    data: Union[cupy.ndarray, _DataPlaceholder]
    ready: Event
    index: tuple[slice, ...]
    updates: list[_PartialUpdate]
    prevent_gc: Any

    def __init__(
        self, data: Union[cupy.ndarray, _DataPlaceholder],
        ready: Event, index: tuple[slice, ...],
        updates: Optional[list[_PartialUpdate]] = None,
        prevent_gc: Any = None,
    ) -> None:
        self.data = data
        self.ready = ready
        self.index = index
        self.updates = updates if updates is not None else []
        self.prevent_gc = prevent_gc

    def copy(self) -> '_Chunk':
        if isinstance(self.data, _DataPlaceholder):
            data = self.data
            ready = self.ready
        else:
            with self.data.device:
                stream = get_current_stream()
                stream.wait_event(self.ready)
                data = self.data.copy()
                self.ready.record(stream)
                ready = stream.record()

        return _Chunk(data, ready, self.index, list(self.updates),
                      prevent_gc=self.prevent_gc)

    def apply_updates(self, mode: _Mode, dtype: numpy.dtype) -> None:
        """Apply all updates in-place."""
        if len(self.updates) == 0:
            return

        with self.data.device:
            stream = cupy.cuda.get_current_stream()
            stream.wait_event(self.ready)

            if isinstance(self.data, _DataPlaceholder):
                if _is_op_mode(mode):
                    value = mode.identity_of(dtype)
                    data = cupy.full(self.data.shape, value, dtype)
                else:
                    data = cupy.empty(self.data.shape, dtype)
                self.data = cupy.atleast_1d(data)

            for new_data, idx in self.updates:
                stream.wait_event(new_data.ready)
                stream.synchronize()
                if _is_op_mode(mode):
                    self.data[idx] = mode.func(self.data[idx], new_data.data)
                else:
                    self.data[idx] = new_data.data
                stream.synchronize()

            self.ready.record(stream)
            self.prevent_gc = (self.prevent_gc, self.updates)
            self.updates = []

        self.ready.synchronize()


if nccl.available:
    def _create_communicators(
        devices: Iterable[int],
    ) -> dict[int, NcclCommunicator]:
        comms_list = NcclCommunicator.initAll(list(devices))
        return {comm.device_id(): comm for comm in comms_list}


    def _transfer(
        src_comm: NcclCommunicator, src_stream: Stream, src_data: _ManagedData,
        dst_comm: NcclCommunicator, dst_stream: Stream, dst_dev: int
    ) -> _DataTransfer:
        src_dev = src_data.data.device.id
        if src_dev == dst_dev:
            return _DataTransfer(src_data.data, src_data.ready)

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
else:
    def _create_communicators(
        devices: Iterable[int],
    ) -> dict[int, NcclCommunicator]:
        return {dev: NcclCommunicator() for dev in devices}


    def _transfer(
        src_comm: NcclCommunicator, src_stream: Stream, src_data: _ManagedData,
        dst_comm: NcclCommunicator, dst_stream: Stream, dst_dev: int
    ) -> _DataTransfer:
        src_dev = src_data.data.device.id
        if src_dev == dst_dev:
            return _DataTransfer(src_data.data, src_data.ready)

        with Device(dst_dev):
            dst_stream.wait_event(src_data.ready)
            with dst_stream:
                dst_data = src_data.data.copy()
            return _DataTransfer(dst_data, dst_stream.record(),
                                 prevent_gc=src_data.data)


class _DistributedArray(cupy.ndarray):
    _chunks_map: dict[int, list[_Chunk]]
    _streams: dict[int, Stream]
    _mode: _Mode
    _comms: dict[int, NcclCommunicator]    # type: ignore
    _mem: cupy.cuda.Memory

    def __new__(
        cls, shape: tuple[int, ...], dtype: Any,
        chunks_map: dict[int, list[_Chunk]],
        mode: _Mode = _REPLICA_MODE,
        comms: Optional[dict[int, NcclCommunicator]]   # type: ignore
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
        else:
            obj._comms = _create_communicators(chunks_map.keys())
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
        for mode_str, mode_obj in _MODES.items():
            if self._mode is mode_obj:
                return mode_str
        raise RuntimeError('Unrecognized mode')

    @property
    def devices(self) -> Iterable[int]:
        return self._chunks_map.keys()

    @property
    def index_map(self) -> dict[int, list[tuple[slice, ...]]]:
        return {dev: [chunk.index for chunk in chunks]
                for dev, chunks in self._chunks_map.items()}

    def _prepare_comms(self, devices: Iterable[int]) -> None:
        devices = self._chunks_map.keys() | devices
        if devices <= self._comms.keys():
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
        chunk.apply_updates(mode, self.dtype)

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
        self, dist_args: list[tuple[Union[int, str], '_DistributedArray']],
        regular_args: list[tuple[Union[int, str], cupy.ndarray]],
        dev: int, chunk_i: int, idx: tuple[slice, ...],
    ) -> list[tuple[Union[int, str], Union[_ManagedData, _DataPlaceholder]]]:
        # Dist arrays must have chunk_map of compatible shapes, otherwise
        # hard error.
        # In case that they are of different, but broadcastable shapes
        # Data movement may be needed
        # Currently: Support only same shape chunk_map
        args: list[tuple[Union[int, str],
                         Union[_ManagedData, _DataPlaceholder]]] = []
        for i, arg in dist_args:
            chunk = arg._chunks_map[dev][chunk_i]
            if isinstance(chunk.data, _DataPlaceholder):
                args.append((i, chunk.data))
            else:
                args.append(
                    (i, _ManagedData(chunk.data, chunk.ready,
                                     chunk.prevent_gc)))

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
        self, dist_args: list[tuple[Union[int, str], '_DistributedArray']],
        dev: int, chunk_i: int,
    ) -> tuple[Optional[Union[int, str]], list[_PartialUpdate]]:
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

        # TODO leave one chunk with updates
        for i, arg in dist_args:
            for chunk in arg._chunks_map[dev]:
                self._apply_updates(chunk, _REPLICA_MODE)
        return None, []

    def _is_peer_access_needed(
        self, dist_args: list[tuple[Union[int, str], '_DistributedArray']],
    ) -> bool:
        index_map = self.index_map
        for _, arg in dist_args:
            if arg.index_map != index_map:
                return True

        return False

    def _execute_kernel_peer_access(
        self, kernel,
        dist_args: list[tuple[Union[int, str], '_DistributedArray']],
        regular_args: list[tuple[Union[int, str], cupy.ndarray]],
    ) -> '_DistributedArray':
        if len(regular_args) > 0:
            raise RuntimeError(
                'Mix `cupy.ndarray` with distributed arrays is currently not'
                ' supported')
        if len(dist_args) > 2:
            raise RuntimeError(
                'Element-wise operation over more than two distributed arrays'
                ' is not supported unless they share the same index_map.')
        args: list[Optional['_DistributedArray']] = [None, None]
        for i, arg in dist_args:
            if isinstance(i, str):
                raise RuntimeError(
                    'Keyword argument is not supported when peer access is'
                    ' necessary in executing an element-wise operation.')
            args[i] = arg

        args = typing.cast(list['_DistributedArray'], args)

        for arg in args:
            for chunks in arg._chunks_map.values():
                for chunk in chunks:
                    self._apply_updates(chunk, _REPLICA_MODE)

        a, b = args

        # TODO: use numpy.result_type and compare
        if isinstance(kernel, _core.ufunc):
            op = kernel._ops._guess_routine_from_in_types((a.dtype, b.dtype))
            if op is None:
                raise RuntimeError(
                    f'Could not guess the return type of {kernel.name}'
                    f' with arguments of type {(a.dtype.type, b.dtype.type)}')
            out_dtypes = op.out_types
        else:
            assert isinstance(kernel, _core._kernel.ElementwiseKernel)
            out_dtypes = kernel._decide_params_type(
                (a.dtype.type, b.dtype.type), ([],))

        if len(out_dtypes) != 1:
            raise RuntimeError(
                'Kernels returning other than signle array are not'
                ' supported')
        dtype = out_dtypes[0]

        chunks_map: dict[int, list[_Chunk]] = {}

        for a_dev, a_chunks in a._chunks_map.items():
            chunks_map[a_dev] = []
            with cupy.cuda.Device(a_dev):
                stream = get_current_stream()

                for a_chunk in a_chunks:
                    assert isinstance(a_chunk.data, cupy.ndarray)

                    new_chunk_data = cupy.empty(a_chunk.data.shape, dtype)
                    stream.wait_event(a_chunk.ready)

                    for b_dev, b_chunks in b._chunks_map.items():
                        for b_chunk in b_chunks:
                            assert isinstance(b_chunk.data, cupy.ndarray)

                            intersection = _index_arith.index_intersection(
                                a_chunk.index, b_chunk.index, self.shape)
                            if intersection is None:
                                continue

                            _core._kernel._check_peer_access(
                                b_chunk.data, a_dev)

                            a_new_idx = _index_arith.index_for_subindex(
                                a_chunk.index, intersection, self.shape)
                            b_new_idx = _index_arith.index_for_subindex(
                                b_chunk.index, intersection, self.shape)

                            # TODO check kernel.nin
                            stream.wait_event(b_chunk.ready)
                            kernel(a_chunk.data[a_new_idx],
                                   b_chunk.data[b_new_idx],
                                   new_chunk_data[a_new_idx])

                    chunks_map[a_dev].append(_Chunk(
                        new_chunk_data, stream.record(), a_chunk.index,
                        updates=[], prevent_gc=b._chunks_map))

        return _DistributedArray(
            self.shape, dtype, chunks_map, _REPLICA_MODE, self._comms)

    def _execute_kernel(
        self, kernel, args: tuple[Any, ...], kwargs: dict[str, Any],
    ) -> '_DistributedArray':
        dist_args: list[tuple[Union[int, str], '_DistributedArray']] = []
        regular_args: list[tuple[Union[int, str], cupy.ndarray]] = []
        i: Union[int, str]
        index_map = self.index_map
        for i, arg in enumerate(args):
            if arg.shape != self.shape:
                # TODO support broadcasting
                raise RuntimeError('Mismatched shapes')

            if isinstance(arg, _DistributedArray):
                dist_args.append((i, arg.to_replica_mode()))
            elif isinstance(arg, cupy.ndarray):
                regular_args.append((i, arg))
            else:
                raise RuntimeError('Unsupported argument type')

        # Do it for kwargs too
        for k, arg in kwargs.items():
            if arg.shape != self.shape:
                # TODO support broadcasting
                raise RuntimeError('Mismatched shapes')
            if arg.index_map != index_map:
                # TODO enable p2p access
                raise RuntimeError('Mismatched index_map')
            if isinstance(arg, _DistributedArray):
                dist_args.append((k, arg))
            elif isinstance(arg, cupy.ndarray):
                regular_args.append((k, arg))
            else:
                raise RuntimeError('Unsupported argument type')

        peer_access = self._is_peer_access_needed(dist_args)
        if peer_access:
            return self._execute_kernel_peer_access(
                kernel, dist_args, regular_args)

        args = list(args)
        new_dtype = None
        new_chunks_map: dict[int, list[_Chunk]] = {}
        for dev, idxs in index_map.items():
            new_chunks_map[dev] = []
            with Device(dev):
                stream = get_current_stream()

                for chunk_i, idx in enumerate(idxs):
                    # This must be called BEFORE self._prepare_args
                    # _prepare_updates may call self._apply_updates
                    # which replaces a placeholder with an actual chunk
                    incoming_index, update_map = self._prepare_updates(
                        dist_args, dev, chunk_i)

                    array_args = self._prepare_args(
                        dist_args, regular_args, dev, chunk_i, idx)

                    placeholder_found = None
                    for i, arg in array_args:
                        if isinstance(arg, _DataPlaceholder):
                            placeholder_found = arg
                            if isinstance(i, int):
                                args[i] = None
                            else:
                                kwargs[i] = None
                            continue
                        stream.wait_event(arg.ready)
                        if isinstance(i, int):
                            args[i] = arg.data
                        else:
                            kwargs[i] = arg.data

                    if placeholder_found:
                        new_chunk = _Chunk(
                            placeholder_found, Event(), index_map[dev][chunk_i])
                        new_chunk.updates = []
                    else:
                        new_data = kernel(*args, **kwargs)

                        new_dtype = new_data.dtype
                        new_chunk = _Chunk(
                            new_data, stream.record(),
                            index_map[dev][chunk_i],
                            prevent_gc=(args, kwargs))

                    new_chunks_map[dev].append(new_chunk)

                    if len(update_map) == 0:
                        continue

                    incoming_index = typing.cast(
                        Union[int, str], incoming_index)

                    args_slice = [None] * len(args)
                    kwargs_slice: dict[str, cupy.ndarray] = {}
                    for update, idx in update_map:
                        for i, arg in enumerate(args):
                            if arg is not None:
                                args_slice[i] = arg[idx]
                        for k, arg in kwargs.items():
                            if arg is not None:
                                kwargs_slice[k] = arg[idx]

                        if isinstance(incoming_index, int):
                            args_slice[incoming_index] = update.data
                        else:
                            kwargs_slice[incoming_index] = update.data

                        stream.wait_event(update.ready)
                        new_data = kernel(*args_slice, **kwargs_slice)
                        new_dtype = new_data.dtype
                        execution_done = stream.record()

                        data_transfer = _DataTransfer(
                            new_data, execution_done,
                            prevent_gc=(args_slice, kwargs_slice))
                        new_chunk.updates.append((data_transfer, idx))

        for chunks in new_chunks_map.values():
            for chunk in chunks:
                if not isinstance(chunk.data, cupy.ndarray) and not isinstance(chunk.data, _DataPlaceholder):
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

    def _transfer(
            self, src_data: _ManagedData, dst_dev: int) -> _DataTransfer:
        src_dev = src_data.data.device.id
        return _transfer(
            self._comms[src_dev], self._stream_for(src_dev), src_data,
            self._comms[dst_dev], self._stream_for(dst_dev), dst_dev)

    def _apply_and_update_chunks(
        self, op_mode: _OpMode, shape: tuple[int, ...],
        src_chunk: _Chunk, dst_chunk: _Chunk,
    ) -> None:
        """Apply `src_chunk` onto `dst_chunk` in `op_mode`.
        There must not be any undone partial update to src_chunk."""
        assert isinstance(src_chunk.data, cupy.ndarray)

        src_dev = src_chunk.data.device.id
        dst_dev = dst_chunk.data.device.id
        src_idx = src_chunk.index
        dst_idx = dst_chunk.index

        intersection = _index_arith.index_intersection(src_idx, dst_idx, shape)
        if intersection is None:
            return
        src_new_idx = _index_arith.index_for_subindex(src_idx, intersection, shape)
        dst_new_idx = _index_arith.index_for_subindex(dst_idx, intersection, shape)

        data_to_transfer = _ManagedData(
            src_chunk.data[src_new_idx], src_chunk.ready, src_chunk.prevent_gc)

        if not op_mode.idempotent:
            with cupy.cuda.Device(src_dev):
                data_to_transfer = data_to_transfer.copy()
            copy_done = data_to_transfer.ready

        update = self._transfer(data_to_transfer, dst_dev)
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
        """There must not be any undone partial update to src_chunk."""

        assert len(src_chunk.updates) == 0
        assert isinstance(src_chunk.data, cupy.ndarray)

        src_idx = src_chunk.index
        dst_idx = dst_chunk.index
        intersection = _index_arith.index_intersection(src_idx, dst_idx, shape)
        if intersection is None:
            return

        dst_dev = dst_chunk.data.device.id
        src_new_idx = _index_arith.index_for_subindex(src_idx, intersection, shape)
        dst_new_idx = _index_arith.index_for_subindex(dst_idx, intersection, shape)


        src_dev = src_chunk.data.device.id
        src_partial_chunk = _ManagedData(
            src_chunk.data[src_new_idx], src_chunk.ready,
            src_chunk.prevent_gc)
        update = self._transfer(src_partial_chunk, dst_dev)
        dst_chunk.updates.append((update, dst_new_idx))

    def _set_identity_on_intersection(
        self, shape: tuple[int, ...], identity,
        a_chunk: _Chunk, b_idx: tuple[slice, ...],
    ) -> None:
        assert isinstance(a_chunk.data, cupy.ndarray)

        a_idx = a_chunk.index
        intersection = _index_arith.index_intersection(a_idx, b_idx, shape)
        if intersection is None:
            return
        a_new_idx = _index_arith.index_for_subindex(a_idx, intersection, shape)
        with a_chunk.data.device:
            stream = get_current_stream()
            stream.wait_event(a_chunk.ready)
            a_chunk.data[a_new_idx] = identity
            stream.record(a_chunk.ready)

    def _set_identity_on_ignored_entries(
            self, identity, chunk: _Chunk) -> None:
        if isinstance(chunk.data, _DataPlaceholder):
            return

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

                    new_index = chunk.index[:axis] + chunk.index[axis+1:]

                    if isinstance(chunk.data, _DataPlaceholder):
                        old_shape = chunk.data.shape
                        new_shape = old_shape[:axis] + old_shape[axis+1:]
                        new_chunk = _Chunk(
                            _DataPlaceholder(new_shape, chunk.data.device),
                            chunk.ready, new_index, [],
                            prevent_gc=chunk.prevent_gc)
                    else:
                        new_data = cupy.atleast_1d(
                            kernel(chunk.data, axis=axis, dtype=dtype))

                        new_dtype = new_data.dtype
                        new_chunk = _Chunk(
                            new_data, execution_stream.record(), new_index, [],
                            prevent_gc=chunk.prevent_gc)

                    new_chunks_map[dev].append(new_chunk)

                    if len(chunk.updates) == 0:
                        continue

                    for update, update_index in chunk.updates:
                        execution_stream.wait_event(update.ready)
                        new_update_data = cupy.atleast_1d(
                            kernel(update.data, axis=axis, dtype=dtype))
                        new_dtype = new_update_data.dtype

                        data_transfer = _DataTransfer(
                            new_update_data, execution_stream.record(),
                            prevent_gc=update.prevent_gc)
                        new_index = update_index[:axis] + update_index[axis+1:]
                        new_chunk.updates.append((data_transfer, new_index))

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
                idxs[i] = _index_arith.normalize_index(self.shape, idxs[i])
            idxs.sort(key=lambda slices:
                      [s.indices(l) for s, l in zip(slices, self.shape)])
            new_index_map[dev] = idxs

        if new_index_map == self.index_map:
            return self

        old_chunks_map = self._chunks_map
        new_chunks_map: dict[int, list[_Chunk]] = {}

        if _is_op_mode(self._mode):
            identity = self._mode.identity_of(self.dtype)

        for dev, idxs in new_index_map.items():
            new_chunks_map[dev] = []

            for idx in idxs:
                with Device(dev):
                    dst_shape = _index_arith.shape_after_indexing(
                        self.shape, idx)
                    stream = get_current_stream()
                    data = _DataPlaceholder(dst_shape, Device(dev))
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
    devices: Optional[Union[int, Iterable[int]]] = None,
    comms: Optional[dict[int, NcclCommunicator]] = None, # type: ignore
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
            array.shape, array.dtype, array._chunks_map, array._mode, comms)

    if not isinstance(array, (numpy.ndarray, cupy.ndarray)):
        array = numpy.array(array)
    elif mode != 'replica':
        array = array.copy()

    new_index_map: dict[int, list[tuple[slice, ...]]] = {}
    for dev, idxs in index_map.items():
        if not isinstance(idxs, list):
            idxs = [idxs]
        for i in range(len(idxs)):
            idxs[i] = _index_arith.normalize_index(array.shape, idxs[i])
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
                if src_array.device.id == dst_dev:
                    return _Chunk(
                        src_data.data, src_data.ready, idx,
                        prevent_gc=src_data.prevent_gc)
                copied = _transfer(
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
