import dataclasses
import typing
from typing import Any, Callable, Final, Iterable, Optional, TypeVar, Union
from typing_extensions import TypeGuard
from cupyx.distributed.array import _chunk
from cupyx.distributed.array._chunk import Chunk
from cupyx.distributed.array import _elementwise
from cupyx.distributed.array import _reduction
from cupyx.distributed.array import _modes
from cupyx.distributed.array import _data_transfer


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
from cupyx.distributed.array import _linalg
from cupyx.distributed.array import _index_arith


class _MultiDeviceDummyMemory(cupy.cuda.Memory):
    pass


class _MultiDeviceDummyPointer(cupy.cuda.MemoryPointer):
    @property
    def device(self) -> Device:
        # This override is needed to assign an invalid device id
        # Since the array is not residing in a single device now
        return Device(-1)


class _DistributedArray(cupy.ndarray):
    _chunks_map: dict[int, list[Chunk]]
    _streams: dict[int, Stream]
    _mode: _modes.Mode
    _comms: dict[int, NcclCommunicator]    # type: ignore
    _mem: cupy.cuda.Memory

    def __new__(
        cls, shape: tuple[int, ...], dtype: Any,
        chunks_map: dict[int, list[Chunk]],
        mode: _modes.Mode = _modes.REPLICA_MODE,
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
            obj._comms = _data_transfer.create_communicators(chunks_map.keys())
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
        for mode_str, mode_obj in _modes.MODES.items():
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
        self._comms = _data_transfer.create_communicators(devices)

    def _stream_for(self, dev: int) -> Stream:
        if dev not in self._streams:
            with Device(dev):
                self._streams[dev] = Stream()

        return self._streams[dev]

    def _apply_updates(self, chunk: Chunk, mode: _modes.Mode) -> None:
        chunk.apply_updates(mode)

    def _apply_updates_all_chunks(self) -> None:
        transfer_events = []

        for chunks in self._chunks_map.values():
            for chunk in chunks:
                self._apply_updates(chunk, self._mode)
                transfer_events.append(chunk.ready)

    def __cupy_override_elementwise_kernel__(self, kernel, *args, **kwargs):
        # This defines a protocol to be called from elementwise kernel
        # to override some of the ops done there
        return _elementwise.execute(kernel, args, kwargs)

    def _transfer(
            self, src_data: _data_transfer.ManagedData, dst_dev: int) -> _data_transfer.DataTransfer:
        src_dev = src_data.data.device.id
        return _data_transfer.transfer(
            self._comms[src_dev], self._stream_for(src_dev), src_data,
            self._comms[dst_dev], self._stream_for(dst_dev), dst_dev)

    def __cupy_override_reduction_kernel__(
            self, kernel, axis, dtype, out, keepdims) -> Any:
        # This defines a protocol to be called from reduction functions
        # to override some of the ops done there
        if axis is None:
            raise RuntimeError('axis must be specified')
        if out is not None:
            raise RuntimeError('Argument `out` is not supported')
        if keepdims:
            raise RuntimeError('Argument `keepdims` is not supported')
        return _reduction.execute(self, kernel, axis, dtype)

    def _copy_chunks_map(self) -> dict[int, list[Chunk]]:
        return {dev: [chunk.copy() for chunk in chunks]
                for dev, chunks in self._chunks_map.items()}

    def _copy_chunks_map_replica_mode(self) -> dict[int, list[Chunk]]:
        """Return a copy of the chunks_map in the replica mode."""
        chunks_map = self._copy_chunks_map()
        if _modes.is_op_mode(self._mode):
            _chunk.all_reduce_intersections(
                self, self._mode, self.shape, chunks_map)
        return chunks_map

    def _replica_mode_chunks_map(self) -> dict[int, list[Chunk]]:
        """Return a view or a copy of the chunks_map in the replica mode."""
        if self._mode is _modes.REPLICA_MODE:
            return self._chunks_map

        if len(self._chunks_map) == 1:
            chunks, = self._chunks_map.values()
            if len(chunks) == 1:
                self._apply_updates(chunks[0], self._mode)
                return self._chunks_map

        return self._copy_chunks_map_replica_mode()

    def to_replica_mode(self) -> '_DistributedArray':
        """Return a view or a copy of self in the replica mode."""
        if self._mode is _modes.REPLICA_MODE:
            return self
        else:
            chunks_map = self._replica_mode_chunks_map()
            return _DistributedArray(
                self.shape, self.dtype, chunks_map, _modes.REPLICA_MODE, self._comms)

    def change_mode(self, mode: str) -> '_DistributedArray':
        """Return a view or a copy of self in the given mode."""
        if mode not in _modes.MODES:
            raise RuntimeError(f'`mode` must be one of {list(_modes.MODES)}')

        mode_obj = _modes.MODES[mode]
        if mode_obj is self._mode:
            return self

        if _modes.is_op_mode(mode_obj):
            chunks_map = _chunk.to_op_mode(self, mode_obj)
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
        new_chunks_map: dict[int, list[Chunk]] = {}

        if _modes.is_op_mode(self._mode):
            identity = self._mode.identity_of(self.dtype)

        for dev, idxs in new_index_map.items():
            new_chunks_map[dev] = []

            for idx in idxs:
                with Device(dev):
                    dst_shape = _index_arith.shape_after_indexing(
                        self.shape, idx)
                    stream = get_current_stream()
                    data = _chunk.DataPlaceholder(dst_shape, Device(dev))
                    new_chunk = Chunk(data, stream.record(), idx)
                    new_chunks_map[dev].append(new_chunk)

        for src_chunks in old_chunks_map.values():
            for src_chunk in src_chunks:
                self._apply_updates(src_chunk, self._mode)

                if _modes.is_op_mode(self._mode):
                    src_chunk = src_chunk.copy()

                for dst_chunks in new_chunks_map.values():
                    for dst_chunk in dst_chunks:
                        if _modes.is_op_mode(self._mode):
                            _chunk.apply_and_update_chunks(
                                self, self._mode, self.shape, src_chunk, dst_chunk)
                        else:
                            _chunk.copy_on_intersection(
                                self, self.shape, src_chunk, dst_chunk)

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
            f: Callable[[Chunk], Chunk],
            chunks_map: dict[int, list[Chunk]],
            ) -> dict[int, list[Chunk]]:
        new_chunks_map = {}
        for dev, chunks in chunks_map.items():
            new_chunks_map[dev] = [f(chunk) for chunk in chunks]
        return new_chunks_map

    def _change_shape(
        self,
        f_shape: Callable[[tuple[int, ...]], tuple[int, ...]],
        f_idx: Callable[[tuple[slice, ...]], tuple[slice, ...]],
    ) -> '_DistributedArray':
        def apply_to_chunk(chunk: Chunk) -> Chunk:
            data = chunk.data.reshape(f_shape(chunk.data.shape))
            index = f_idx(chunk.index)
            updates = [(data, f_idx(idx))
                       for data, idx in chunk.updates]
            return Chunk(data, chunk.ready, index, updates, chunk.prevent_gc)

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

        if _modes.is_op_mode(self._mode):
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
                if _modes.is_op_mode(self._mode):
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

        comms = _data_transfer.create_communicators(devices)

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
                src_data = _data_transfer.ManagedData(src_array, src_stream.record(),
                                        prevent_gc=src_array)
            with Device(dst_dev):
                dst_stream = get_current_stream()
                if src_array.device.id == dst_dev:
                    return Chunk(
                        src_data.data, src_data.ready, idx,
                        prevent_gc=src_data.prevent_gc)
                copied = _data_transfer.transfer(
                    comms[src_dev], src_stream, src_data,
                    comms[dst_dev], dst_stream, dst_dev)
                return Chunk(copied.data, copied.ready, idx,
                              prevent_gc=src_data)

    else:
        def make_chunk(dev, idx, array):
            with Device(dev):
                stream = get_current_stream()
                copied = cupy.array(array)
                return Chunk(copied, stream.record(), idx,
                              prevent_gc=array)

    mode_obj = _modes.MODES[mode]
    chunks_map: dict[int, list[Chunk]] = {}
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
