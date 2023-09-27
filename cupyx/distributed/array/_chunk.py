from typing import Any, Optional, Union
from typing_extensions import TypeGuard

import dataclasses
from itertools import chain

import numpy

from cupy._core.core import ndarray
import cupy._creation.basic as _creation_basic
import cupy._manipulation.dims as _manipulation_dims
from cupy.cuda.device import Device
from cupy.cuda.stream import Event
from cupy.cuda.stream import Stream
from cupy.cuda.stream import get_current_stream

from cupyx.distributed.array import _modes
from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array._data_transfer import _NcclCommunicator


class _DataPlaceholder:
    # Mocks ndarray
    # Eventually overwritten by PartialUpdates entirely, so
    # any operation on _DataPlaceholder can be skipped
    shape: tuple[int, ...]
    device: Device

    def __init__(self, shape: tuple[int, ...], device: Device) -> None:
        self.shape = shape
        self.device = device

    def reshape(self, new_shape: tuple[int, ...]) -> '_DataPlaceholder':
        return _DataPlaceholder(new_shape, self.device)

    def to_ndarray(
            self, mode: '_modes._Mode', dtype: numpy.dtype) -> ndarray:
        with self.device:
            if _modes._is_op_mode(mode):
                value = mode.identity_of(dtype)
                data = _creation_basic.full(self.shape, value, dtype)
            else:
                data = _creation_basic.empty(self.shape, dtype)

            # We avoid 0D array because we expect data[idx] to return a view
            return _manipulation_dims.atleast_1d(data)


class _Chunk:
    data: Union[ndarray, _DataPlaceholder]
    ready: Event
    index: tuple[slice, ...]
    _updates: list[_data_transfer._PartialUpdate] = dataclasses.field(
        default_factory=list)
    _prevent_gc: Any = None     # TODO: Release it to avoid OOM

    # Rule: whenever data is DataPlaceholder, ready is empty

    def __init__(
        self, data: Union[ndarray, _DataPlaceholder], ready: Event,
        index: tuple[slice, ...],
        updates: Optional[list[_data_transfer._PartialUpdate]] = None,
        prevent_gc: Any = None
    ) -> None:
        self.data = data
        self.ready = ready
        self.index = index
        self._updates = updates if updates is not None else []
        self._prevent_gc = prevent_gc

    @classmethod
    def create_placeholder(
        cls, shape: tuple[int, ...], device: Union[int, Device],
        index: tuple[slice, ...],
        updates: Optional[list[_data_transfer._PartialUpdate]] = None
    ) -> '_Chunk':
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

    def apply_updates(self, mode: '_modes._Mode') -> None:
        """Apply all updates in-place."""
        if len(self._updates) == 0:
            return

        if isinstance(self.data, _DataPlaceholder):
            dtype = self._updates[0][0].data.dtype
            self.data = self.data.to_ndarray(mode, dtype)

        with self.data.device:
            stream = get_current_stream()
            stream.wait_event(self.ready)

            for update_data, idx in self._updates:
                stream.wait_event(update_data.ready)
                if _modes._is_op_mode(mode):
                    self.data[idx] = mode.func(
                        self.data[idx], update_data.data)
                else:
                    self.data[idx] = update_data.data

            self.ready.record(stream)
            self._prevent_gc = (self._prevent_gc, self._updates)
            self._updates = []

    def apply_to(
        self, target: '_Chunk', mode: '_modes._Mode',
        shape: tuple[int, ...],
        comms: dict[int, _data_transfer._NcclCommunicator],
        streams: dict[int, Stream],
    ) -> None:
        # Overwrite target with mode.func(self, target) on their overlaps
        # This is just appending part of self to target.updates in the mode
        src_chunk = self
        dst_chunk = target

        assert len(src_chunk._updates) == 0
        assert isinstance(src_chunk.data, ndarray)

        src_dev = src_chunk.data.device.id
        dst_dev = dst_chunk.data.device.id
        src_idx = src_chunk.index
        dst_idx = dst_chunk.index

        intersection = _index_arith._index_intersection(
            src_idx, dst_idx, shape)
        if intersection is None:
            return

        src_new_idx = _index_arith._index_for_subindex(
            src_idx, intersection, shape)
        dst_new_idx = _index_arith._index_for_subindex(
            dst_idx, intersection, shape)

        data_to_transfer = _data_transfer._AsyncData(
            src_chunk.data[src_new_idx], src_chunk.ready,
            src_chunk._prevent_gc)

        def is_not_idempotent(mode: _modes._Mode) -> TypeGuard[_modes._OpMode]:
            return mode is not _modes._REPLICA_MODE and not mode.idempotent

        if is_not_idempotent(mode):
            with Device(src_dev):
                data_to_transfer = data_to_transfer.copy()
            copy_done = data_to_transfer.ready

        update = _data_transfer._transfer(
            comms[src_dev], streams[src_dev], data_to_transfer,
            comms[dst_dev], streams[dst_dev], dst_dev)
        dst_chunk.add_update(update, dst_new_idx)

        if is_not_idempotent(mode):
            dtype = src_chunk.data.dtype
            with Device(src_dev):
                stream = get_current_stream()
                stream.wait_event(copy_done)
                src_chunk.data[src_new_idx] = mode.identity_of(dtype)
                stream.record(src_chunk.ready)

    def set_identity_on_intersection(
        self, idx: tuple[slice, ...], shape: tuple[int, ...], identity,
    ) -> None:
        assert isinstance(self.data, ndarray)

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

    def set_identity_on_overwritten_entries(self, identity) -> None:
        if isinstance(self.data, _DataPlaceholder):
            return

        with self.data.device:
            stream = get_current_stream()
            stream.wait_event(self.ready)
            for _, idx in self._updates:
                self.data[idx] = identity
            stream.record(self.ready)


def _all_reduce_intersections(
    op_mode: '_modes._OpMode', shape: tuple[int, ...],
    chunk_map: dict[int, list[_Chunk]],
    comms: dict[int, _NcclCommunicator], streams: dict[int, Stream],
) -> None:
    chunks_list = list(chain.from_iterable(chunk_map.values()))

    # TODO flatten this loop somehow
    for i in range(len(chunks_list)):
        src_chunk = chunks_list[i]
        src_chunk.apply_updates(op_mode)

        for j in range(i + 1, len(chunks_list)):
            dst_chunk = chunks_list[j]

            src_chunk.apply_to(dst_chunk, op_mode, shape, comms, streams)

    for j in range(len(chunks_list) - 1, -1, -1):
        src_chunk = chunks_list[j]
        src_chunk.apply_updates(_modes._REPLICA_MODE)

        for i in range(j):
            dst_chunk = chunks_list[i]
            src_chunk.apply_to(
                dst_chunk, _modes._REPLICA_MODE, shape, comms, streams)
