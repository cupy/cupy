from typing import Any, Optional, Type, TypeVar, Union

import dataclasses

import cupy
from cupy.cuda import Device, Event, get_current_stream

from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array import _modes
from cupyx.distributed.array import _chunk


@dataclasses.dataclass
class DataPlaceholder:
    """Mock cupy.ndarray."""
    shape: tuple[int, ...]
    device: Device

    def copy(self) -> 'DataPlaceholder':
        return self

    def reshape(self, new_shape: tuple[int, ...]) -> 'DataPlaceholder':
        return DataPlaceholder(new_shape, self.device)


@dataclasses.dataclass
class Chunk:
    data: Union[cupy.ndarray, DataPlaceholder]
    ready: Event
    index: tuple[slice, ...]
    updates: list[_data_transfer.PartialUpdate] = dataclasses.field(
        default_factory=list)
    prevent_gc: Any = None

    # Rule: isinstance(data, DataPlaceholder) ==> ready is empty

    @classmethod
    def create_placeholder(
            cls, shape: tuple[int, ...], device: Union[int, Device],
            index: tuple[slice, ...],
            updates: Optional[list[_data_transfer.PartialUpdate]] = None) -> 'Chunk':
        if isinstance(device, int):
            device = Device(device)
        data = DataPlaceholder(shape, device)

        if updates is None:
            updates = []

        return Chunk(data, Event(), index, updates)

    def copy(self) -> 'Chunk':
        if isinstance(self.data, DataPlaceholder):
            data = self.data
            ready = self.ready
        else:
            with self.data.device:
                stream = get_current_stream()
                stream.wait_event(self.ready)
                data = self.data.copy()
                self.ready.record(stream)
                ready = stream.record()

        return Chunk(data, ready, self.index, list(self.updates),
                      prevent_gc=self.prevent_gc)

    def apply_updates(self, mode: _modes.Mode) -> None:
        """Apply all updates in-place."""
        if len(self.updates) == 0:
            return

        with self.data.device:
            stream = cupy.cuda.get_current_stream()
            stream.wait_event(self.ready)

            if isinstance(self.data, DataPlaceholder):
                dtype = self.updates[0][0].data.dtype

                if _modes.is_op_mode(mode):
                    value = mode.identity_of(dtype)
                    data = cupy.full(self.data.shape, value, dtype)
                else:
                    data = cupy.empty(self.data.shape, dtype)

                # We avoid 0D array because we expect data[idx] to return a view
                self.data = cupy.atleast_1d(data)

            for update_data, idx in self.updates:
                stream.wait_event(update_data.ready)
                stream.synchronize()
                if _modes.is_op_mode(mode):
                    self.data[idx] = mode.func(self.data[idx], update_data.data)
                else:
                    self.data[idx] = update_data.data
                stream.synchronize()

            self.ready.record(stream)
            self.prevent_gc = (self.prevent_gc, self.updates)
            self.updates = []

        self.ready.synchronize()


def apply_and_update_chunks(
    self, op_mode: _modes.OpMode, shape: tuple[int, ...],
    src_chunk: Chunk, dst_chunk: Chunk,
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

    data_to_transfer = _data_transfer.ManagedData(
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

def all_reduce_intersections(
    self, op_mode: _modes.OpMode, shape: tuple[int, ...],
    chunk_map: dict[int, list[Chunk]],
) -> None:
    chunks_list = [chunk for chunks in chunk_map.values()
                            for chunk in chunks]

    # TODO flatten this loop somehow
    for i in range(len(chunks_list)):
        src_chunk = chunks_list[i]
        self._apply_updates(src_chunk, op_mode)

        for j in range(i + 1, len(chunks_list)):
            dst_chunk = chunks_list[j]
            apply_and_update_chunks(
                self, op_mode, shape, src_chunk, dst_chunk)

    for j in range(len(chunks_list) - 1, -1, -1):
        src_chunk = chunks_list[j]
        self._apply_updates(src_chunk, _modes.REPLICA_MODE)

        for i in range(j):
            dst_chunk = chunks_list[i]
            copy_on_intersection(self, shape, src_chunk, dst_chunk)

def copy_on_intersection(
    self, shape: tuple[int, ...],
    src_chunk: Chunk, dst_chunk: Chunk,
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
    src_partial_chunk = _data_transfer.ManagedData(
        src_chunk.data[src_new_idx], src_chunk.ready,
        src_chunk.prevent_gc)
    update = self._transfer(src_partial_chunk, dst_dev)
    dst_chunk.updates.append((update, dst_new_idx))

def set_identity_on_intersection(
    self, shape: tuple[int, ...], identity,
    a_chunk: Chunk, b_idx: tuple[slice, ...],
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

def set_identity_on_ignored_entries(
        self, identity, chunk: Chunk) -> None:
    if isinstance(chunk.data, DataPlaceholder):
        return

    with chunk.data.device:
        stream = get_current_stream()
        stream.wait_event(chunk.ready)
        for _, idx in chunk.updates:
            chunk.data[idx] = identity
        stream.record(chunk.ready)

def to_op_mode(self, op_mode: _modes.OpMode) -> dict[int, list[Chunk]]:
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
            self._apply_updates(chunk, _modes.REPLICA_MODE)

    chunks_list = [chunk for chunks in chunks_map.values()
                            for chunk in chunks]
    identity = op_mode.identity_of(self.dtype)

    # TODO: Parallelize
    for i in range(len(chunks_list)):
        a_chunk = chunks_list[i]
        for j in range(i + 1, len(chunks_list)):
            b_chunk = chunks_list[j]
            _chunk.set_identity_on_intersection(
                self, self.shape, identity, a_chunk, b_chunk.index)

    return chunks_map
