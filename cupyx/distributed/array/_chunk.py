import contextlib
from itertools import chain
from typing import Any, Iterator, Optional, Union

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
from cupyx.distributed.array._data_transfer import _Communicator


class _ArrayPlaceholder:
    # Mocks ndarray
    # Eventually overwritten by PartialUpdates entirely, so
    # any operation on _DataPlaceholder can be skipped
    shape: tuple[int, ...]
    device: Device

    def __init__(self, shape: tuple[int, ...], device: Device) -> None:
        self.shape = shape
        self.device = device

    def reshape(self, new_shape: tuple[int, ...]) -> '_ArrayPlaceholder':
        return _ArrayPlaceholder(new_shape, self.device)

    def to_ndarray(
            self, mode: '_modes.Mode', dtype: numpy.dtype) -> ndarray:
        with self.device:
            if mode is _modes.REPLICA:
                data = _creation_basic.empty(self.shape, dtype)
            else:
                value = mode.identity_of(dtype)
                data = _creation_basic.full(self.shape, value, dtype)

            # We avoid 0D array because we expect data[idx] to return a view
            return _manipulation_dims.atleast_1d(data)


class _Chunk:
    array: Union[ndarray, _ArrayPlaceholder]
    ready: Event
    index: tuple[slice, ...]
    updates: list[_data_transfer._PartialUpdate]
    prevent_gc: Any = None     # TODO: Release it to avoid OOM

    # Rule: whenever data is DataPlaceholder, ready is empty

    def __init__(
        self, data: Union[ndarray, _ArrayPlaceholder], ready: Event,
        index: tuple[slice, ...],
        updates: Optional[list[_data_transfer._PartialUpdate]] = None,
        prevent_gc: Any = None
    ) -> None:
        self.array = data
        self.ready = ready
        self.index = index
        self.updates = updates if updates is not None else []
        self.prevent_gc = prevent_gc

    @classmethod
    def create_placeholder(
        cls, shape: tuple[int, ...], device: Union[int, Device],
        index: tuple[slice, ...],
        updates: Optional[list[_data_transfer._PartialUpdate]] = None,
    ) -> '_Chunk':
        if isinstance(device, int):
            device = Device(device)

        data = _ArrayPlaceholder(shape, device)
        with device:
            ready = Event()
        if updates is None:
            updates = []

        return _Chunk(data, ready, index, updates)

    @contextlib.contextmanager
    def on_ready(self) -> Iterator[Stream]:
        with self.array.device:
            stream = get_current_stream()
            stream.wait_event(self.ready)
            yield stream

    def add_update(
        self, update: _data_transfer._AsyncData, idx: tuple[slice, ...],
    ) -> None:
        self.updates.append((update, idx))

    def copy(self) -> '_Chunk':
        # TODO: Calling flush here would reduce the amount of future copying
        if isinstance(self.array, _ArrayPlaceholder):
            data = self.array
            ready = self.ready
        else:
            with self.on_ready() as stream:
                data = self.array.copy()
                ready = stream.record()

        return _Chunk(data, ready, self.index, list(self.updates),
                      prevent_gc=self.prevent_gc)

    def flush(self, mode: '_modes.Mode') -> None:
        """Apply all updates in-place."""
        if len(self.updates) == 0:
            return

        if isinstance(self.array, _ArrayPlaceholder):
            dtype = self.updates[0][0].array.dtype
            self.array = self.array.to_ndarray(mode, dtype)

        with self.on_ready() as stream:
            for update_data, idx in self.updates:
                stream.wait_event(update_data.ready)
                if mode is _modes.REPLICA:
                    self.array[idx] = update_data.array
                else:
                    self.array[idx] = mode.func(
                        self.array[idx], update_data.array)

            stream.record(self.ready)
            self.prevent_gc = (self.prevent_gc, self.updates)
            self.updates = []

    def apply_to(
        self, target: '_Chunk', mode: '_modes.Mode',
        shape: tuple[int, ...],
        comms: dict[int, _data_transfer._Communicator],
        streams: dict[int, Stream],
    ) -> None:
        # Overwrite target with mode.func(self, target) on their overlaps
        # This is just appending part of self to target.updates in the mode
        src_chunk = self
        dst_chunk = target

        assert len(src_chunk.updates) == 0
        assert isinstance(src_chunk.array, ndarray)

        src_dev = src_chunk.array.device.id
        dst_dev = dst_chunk.array.device.id
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
            src_chunk.array[src_new_idx], src_chunk.ready,
            src_chunk.prevent_gc)

        if mode is not _modes.REPLICA and not mode.idempotent:
            data_to_transfer = data_to_transfer.copy()

        update = _data_transfer._transfer(
            comms[src_dev], streams[src_dev], data_to_transfer,
            comms[dst_dev], streams[dst_dev], dst_dev)
        dst_chunk.add_update(update, dst_new_idx)

        if mode is not _modes.REPLICA and not mode.idempotent:
            dtype = src_chunk.array.dtype
            with data_to_transfer.on_ready() as stream:
                # Now src data has been copied, so we can write on src_chunk
                src_chunk.array[src_new_idx] = mode.identity_of(dtype)
                stream.record(src_chunk.ready)

    def set_identity_on_intersection(
        self, idx: tuple[slice, ...], shape: tuple[int, ...], identity,
    ) -> None:
        assert isinstance(self.array, ndarray)

        intersection = _index_arith._index_intersection(self.index, idx, shape)
        if intersection is None:
            return
        self_new_idx = _index_arith._index_for_subindex(
            self.index, intersection, shape)
        with self.on_ready() as stream:
            self.array[self_new_idx] = identity
            stream.record(self.ready)

    def set_identity_on_overwritten_entries(self, identity) -> None:
        if isinstance(self.array, _ArrayPlaceholder):
            return

        with self.on_ready() as stream:
            for _, idx in self.updates:
                self.array[idx] = identity
            stream.record(self.ready)


def _all_reduce_intersections(
    op_mode: '_modes._OpMode', shape: tuple[int, ...],
    chunk_map: dict[int, list[_Chunk]],
    comms: dict[int, _Communicator], streams: dict[int, Stream],
) -> None:
    chunks_list = list(chain.from_iterable(chunk_map.values()))

    for i in range(len(chunks_list)):
        src_chunk = chunks_list[i]
        src_chunk.flush(op_mode)

        for j in range(i + 1, len(chunks_list)):
            dst_chunk = chunks_list[j]

            src_chunk.apply_to(dst_chunk, op_mode, shape, comms, streams)

    for j in range(len(chunks_list) - 1, -1, -1):
        src_chunk = chunks_list[j]
        src_chunk.flush(_modes.REPLICA)

        for i in range(j):
            dst_chunk = chunks_list[i]
            src_chunk.apply_to(
                dst_chunk, _modes.REPLICA, shape, comms, streams)
