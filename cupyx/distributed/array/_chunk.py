from typing import Any, Optional, Type, TypeVar, Union

import dataclasses

import cupy
from cupy.cuda import Device, Event, Stream, get_current_stream

import cupyx.distributed.array as darray
from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array._data_transfer import _NcclCommunicator
from cupyx.distributed.array import _chunk


@dataclasses.dataclass
class _DataPlaceholder:
    """Mock cupy.ndarray."""
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
    updates: list[_data_transfer._PartialUpdate] = dataclasses.field(
        default_factory=list)
    prevent_gc: Any = None

    # Rule: whenever data is DataPlaceholder, ready is empty

    @classmethod
    def create_placeholder(
            cls, shape: tuple[int, ...], device: Union[int, Device],
            index: tuple[slice, ...],
            updates: Optional[list[_data_transfer._PartialUpdate]] = None) -> '_Chunk':
        if isinstance(device, int):
            device = Device(device)
        data = _DataPlaceholder(shape, device)

        if updates is None:
            updates = []

        return _Chunk(data, Event(), index, updates)

    def copy(self) -> '_Chunk':
        if isinstance(self.data, _DataPlaceholder):
            data = self.data
            ready = self.ready
        else:
            with self.data.device:
                stream = get_current_stream()
                stream.wait_event(self.ready)
                data = self.data.copy()
                ready = stream.record()

        return _Chunk(data, ready, self.index, list(self.updates),
                      prevent_gc=self.prevent_gc)

    def _ensure_data_initialized(self, mode: 'darray._Mode') -> None:
        if isinstance(self.data, cupy.ndarray):
            return

        dtype = self.updates[0][0].data.dtype

        with self.data.device:
            if darray._is_op_mode(mode):
                value = mode.identity_of(dtype)
                data = cupy.full(self.data.shape, value, dtype)
            else:
                data = cupy.empty(self.data.shape, dtype)

            # We avoid 0D array because we expect data[idx] to return a view
            self.data = cupy.atleast_1d(data)

    def apply_updates(self, mode: 'darray._Mode') -> None:
        """Apply all updates in-place."""
        if len(self.updates) == 0:
            return

        self._ensure_data_initialized(mode)

        with self.data.device:
            stream = cupy.cuda.get_current_stream()
            stream.wait_event(self.ready)

            for update_data, idx in self.updates:
                stream.wait_event(update_data.ready)
                if darray._is_op_mode(mode):
                    self.data[idx] = mode.func(self.data[idx], update_data.data)
                else:
                    self.data[idx] = update_data.data

            self.ready.record(stream)
            self.prevent_gc = (self.prevent_gc, self.updates)
            self.updates = []


def _apply_chunks(
    new_op_mode: 'darray._OpMode', shape: tuple[int, ...],
    src_chunk: _Chunk, dst_chunk: _Chunk,
    comms: dict[int, _data_transfer._NcclCommunicator],
    streams: dict[int, Stream],
) -> None:
    """Apply `src_chunk` onto `dst_chunk` in `new_op_mode`.
    There must not be any undone partial update to src_chunk."""
    assert isinstance(src_chunk.data, cupy.ndarray)

    src_dev = src_chunk.data.device.id
    dst_dev = dst_chunk.data.device.id
    src_idx = src_chunk.index
    dst_idx = dst_chunk.index

    intersection = _index_arith._index_intersection(src_idx, dst_idx, shape)
    if intersection is None:
        return
    src_new_idx = _index_arith._index_for_subindex(src_idx, intersection, shape)
    dst_new_idx = _index_arith._index_for_subindex(dst_idx, intersection, shape)

    data_to_transfer = _data_transfer._AsyncData(
        src_chunk.data[src_new_idx], src_chunk.ready, src_chunk.prevent_gc)

    if not new_op_mode.idempotent:
        with cupy.cuda.Device(src_dev):
            data_to_transfer = data_to_transfer.copy()
        copy_done = data_to_transfer.ready

    update = _data_transfer._transfer(
        comms[src_dev], streams[src_dev], data_to_transfer,
        comms[dst_dev], streams[dst_dev], dst_dev)
    dst_chunk.updates.append((update, dst_new_idx))

    if not new_op_mode.idempotent:
        dtype = src_chunk.data.dtype
        with Device(src_dev):
            stream = get_current_stream()
            stream.wait_event(copy_done)
            src_chunk.data[src_new_idx] = new_op_mode.identity_of(dtype)
            stream.record(src_chunk.ready)


def _all_reduce_intersections(
    new_op_mode: 'darray._OpMode', shape: tuple[int, ...],
    chunk_map: dict[int, list[_Chunk]],
    comms: dict[int, _NcclCommunicator], streams: dict[int, Stream],
) -> None:
    chunks_list = [chunk for chunks in chunk_map.values()
                            for chunk in chunks]

    # TODO flatten this loop somehow
    for i in range(len(chunks_list)):
        src_chunk = chunks_list[i]
        src_dev = src_chunk.data.device.id
        src_chunk.apply_updates(new_op_mode)

        for j in range(i + 1, len(chunks_list)):
            dst_chunk = chunks_list[j]
            dst_dev = dst_chunk.data.device.id

            _apply_chunks(
                new_op_mode, shape, src_chunk, dst_chunk, comms, streams)

    for j in range(len(chunks_list) - 1, -1, -1):
        src_chunk = chunks_list[j]
        src_chunk.apply_updates(darray._REPLICA_MODE)

        for i in range(j):
            dst_chunk = chunks_list[i]
            _copy_on_intersection(shape, src_chunk, dst_chunk, comms, streams)


def _copy_on_intersection(
    shape: tuple[int, ...],
    src_chunk: _Chunk, dst_chunk: _Chunk,
    comms: dict[int, _NcclCommunicator], streams: dict[int, Stream],
) -> None:
    """There must not be any undone partial update to src_chunk."""

    assert len(src_chunk.updates) == 0
    assert isinstance(src_chunk.data, cupy.ndarray)

    src_idx = src_chunk.index
    dst_idx = dst_chunk.index
    intersection = _index_arith._index_intersection(src_idx, dst_idx, shape)
    if intersection is None:
        return

    src_dev = src_chunk.data.device.id
    dst_dev = dst_chunk.data.device.id
    src_new_idx = _index_arith._index_for_subindex(src_idx, intersection, shape)
    dst_new_idx = _index_arith._index_for_subindex(dst_idx, intersection, shape)

    src_partial_chunk = _data_transfer._AsyncData(
        src_chunk.data[src_new_idx], src_chunk.ready,
        src_chunk.prevent_gc)

    update = _data_transfer._transfer(
        comms[src_dev], streams[src_dev], src_partial_chunk,
        comms[dst_dev], streams[dst_dev], dst_dev)
    dst_chunk.updates.append((update, dst_new_idx))


def _set_identity_on_intersection(
    shape: tuple[int, ...], identity,
    a_chunk: _Chunk, b_idx: tuple[slice, ...],
) -> None:
    assert isinstance(a_chunk.data, cupy.ndarray)

    a_idx = a_chunk.index
    intersection = _index_arith._index_intersection(a_idx, b_idx, shape)
    if intersection is None:
        return
    a_new_idx = _index_arith._index_for_subindex(a_idx, intersection, shape)
    with a_chunk.data.device:
        stream = get_current_stream()
        stream.wait_event(a_chunk.ready)
        a_chunk.data[a_new_idx] = identity
        stream.record(a_chunk.ready)


def _set_identity_on_overwritten_entries(identity, chunk: _Chunk) -> None:
    if isinstance(chunk.data, _DataPlaceholder):
        return

    with chunk.data.device:
        stream = get_current_stream()
        stream.wait_event(chunk.ready)
        for _, idx in chunk.updates:
            chunk.data[idx] = identity
        stream.record(chunk.ready)
