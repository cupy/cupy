from typing import Any, Optional, Type, TypeVar, Union

import dataclasses

import cupy
from cupy.cuda import Device, Event, Stream, get_current_stream

import cupyx.distributed.array as darray
from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array._data_transfer import _NcclCommunicator


@dataclasses.dataclass
class _DataPlaceholder:
    """Mock cupy.ndarray."""
    shape: tuple[int, ...]
    device: Device

    def copy(self) -> '_DataPlaceholder':
        return self

    def reshape(self, new_shape: tuple[int, ...]) -> '_DataPlaceholder':
        return _DataPlaceholder(new_shape, self.device)


class _Chunk:
    data: Union[cupy.ndarray, _DataPlaceholder]
    ready: Event
    index: tuple[slice, ...]
    _updates: list[_data_transfer._PartialUpdate] = dataclasses.field(
        default_factory=list)
    _prevent_gc: Any = None

    # Rule: whenever data is DataPlaceholder, ready is empty

    def __init__(
            self, data: Union[cupy.ndarray, _DataPlaceholder], ready: Event,
            index: tuple[slice, ...],
            updates: Optional[list[_data_transfer._PartialUpdate]] = None,
            prevent_gc: Any = None) -> None:
        self.data = data
        self.ready = ready
        self.index = index
        self._updates = updates if updates is not None else []
        self._prevent_gc = prevent_gc

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

    @property
    def updates(self) -> list[_data_transfer._PartialUpdate]:
        return self._updates

    def add_update(
        self, update: _data_transfer._AsyncData, idx: tuple[slice, ...],
    ) -> None:
        self._updates.append((update, idx))

    def copy(self) -> '_Chunk':
        # TODO: Calling apply_updates here would reduce the amount of
        # future copying
        if isinstance(self.data, _DataPlaceholder):
            data = self.data
            ready = self.ready
        else:
            with self.data.device:
                stream = get_current_stream()
                stream.wait_event(self.ready)
                data = self.data.copy()
                ready = stream.record()

        return _Chunk(data, ready, self.index, list(self._updates),
                      prevent_gc=self._prevent_gc)

    def _ensure_data_initialized(self, mode: 'darray._Mode') -> None:
        if isinstance(self.data, cupy.ndarray):
            return

        dtype = self._updates[0][0].data.dtype

        with self.data.device:
            if darray._is_op_mode(mode):
                value = mode.identity_of(dtype)
                data = cupy.full(self.data.shape, value, dtype)
            else:
                data = cupy.empty(self.data.shape, dtype)

            # We avoid 0D array because we expect data[idx] to return a view
            self.data = cupy.atleast_1d(data)

    def apply_updates(self, mode: 'darray._Mode') -> None:
        """Apply all _updates in-place."""
        if len(self._updates) == 0:
            return

        self._ensure_data_initialized(mode)

        with self.data.device:
            stream = cupy.cuda.get_current_stream()
            stream.wait_event(self.ready)

            for update_data, idx in self._updates:
                stream.wait_event(update_data.ready)
                if darray._is_op_mode(mode):
                    self.data[idx] = mode.func(self.data[idx], update_data.data)
                else:
                    self.data[idx] = update_data.data

            self.ready.record(stream)
            self._prevent_gc = (self._prevent_gc, self._updates)
            self._updates = []


    def _apply_to(
        self, dst_chunk: '_Chunk', new_op_mode: 'darray._OpMode',
        shape: tuple[int, ...],
        comms: dict[int, _data_transfer._NcclCommunicator],
        streams: dict[int, Stream],
    ) -> None:
        assert isinstance(self.data, cupy.ndarray)

        src_chunk = self
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
            src_chunk.data[src_new_idx], src_chunk.ready, src_chunk._prevent_gc)

        if not new_op_mode.idempotent:
            with cupy.cuda.Device(src_dev):
                data_to_transfer = data_to_transfer.copy()
            copy_done = data_to_transfer.ready

        update = _data_transfer._transfer(
            comms[src_dev], streams[src_dev], data_to_transfer,
            comms[dst_dev], streams[dst_dev], dst_dev)
        dst_chunk.add_update(update, dst_new_idx)

        if not new_op_mode.idempotent:
            dtype = src_chunk.data.dtype
            with Device(src_dev):
                stream = get_current_stream()
                stream.wait_event(copy_done)
                src_chunk.data[src_new_idx] = new_op_mode.identity_of(dtype)
                stream.record(src_chunk.ready)

    def _copy_on_intersection(
        self, dst_chunk: '_Chunk', shape: tuple[int, ...],
        comms: dict[int, _NcclCommunicator], streams: dict[int, Stream],
    ) -> None:
        src_chunk = self

        assert len(src_chunk._updates) == 0
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
            src_chunk.data[src_new_idx], src_chunk.ready, src_chunk._prevent_gc)

        update = _data_transfer._transfer(
            comms[src_dev], streams[src_dev], src_partial_chunk,
            comms[dst_dev], streams[dst_dev], dst_dev)
        dst_chunk.add_update(update, dst_new_idx)


    def _set_identity_on_intersection(
        self, idx: tuple[slice, ...], shape: tuple[int, ...], identity,
    ) -> None:
        assert isinstance(self.data, cupy.ndarray)

        intersection = _index_arith._index_intersection(self.index, idx, shape)
        if intersection is None:
            return
        self_new_idx = _index_arith._index_for_subindex(
            self.index, intersection, shape)
        with self.data.device:
            stream = get_current_stream()
            stream.wait_event(self.ready)
            self.data[self_new_idx] = identity
            stream.record(self.ready)

    def _set_identity_on_overwritten_entries(self, identity) -> None:
        if isinstance(self.data, _DataPlaceholder):
            return

        with self.data.device:
            stream = get_current_stream()
            stream.wait_event(self.ready)
            for _, idx in self._updates:
                self.data[idx] = identity
            stream.record(self.ready)


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
        src_chunk.apply_updates(new_op_mode)

        for j in range(i + 1, len(chunks_list)):
            dst_chunk = chunks_list[j]

            src_chunk._apply_to(dst_chunk, new_op_mode, shape, comms, streams)

    for j in range(len(chunks_list) - 1, -1, -1):
        src_chunk = chunks_list[j]
        src_chunk.apply_updates(darray._REPLICA_MODE)

        for i in range(j):
            dst_chunk = chunks_list[i]
            src_chunk._copy_on_intersection(dst_chunk, shape, comms, streams)

