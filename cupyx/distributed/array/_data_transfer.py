import contextlib
import dataclasses
from typing import Any, Iterable, Iterator

from cupy._core.core import ndarray
import cupy._creation.from_data as _creation_from_data
import cupy._creation.basic as _creation_basic
from cupy.cuda.device import Device
from cupy.cuda.stream import Event
from cupy.cuda.stream import Stream
from cupy.cuda.stream import get_current_stream

from cupy.cuda import nccl
from cupyx.distributed._nccl_comm import _get_nccl_dtype_and_count

if nccl.available:
    from cupy.cuda.nccl import NcclCommunicator as _Communicator
else:
    class _MockCommunicator:
        pass

    _Communicator = _MockCommunicator


@dataclasses.dataclass
class _AsyncData:
    array: ndarray
    ready: Event
    prevent_gc: Any = None      # TODO: Release it to avoid OOM

    def copy(self) -> '_AsyncData':
        with self.on_ready() as stream:
            array = self.array.copy()
            stream.record(self.ready)

            return _AsyncData(array, stream.record(), self.prevent_gc)

    @contextlib.contextmanager
    def on_ready(self) -> Iterator[Stream]:
        with self.array.device:
            stream = get_current_stream()
            stream.wait_event(self.ready)
            yield stream


# Overwrite in replica mode, apply in op mode
_PartialUpdate = tuple[_AsyncData, tuple[slice, ...]]


if nccl.available:
    def _create_communicators(
        devices: Iterable[int],
    ) -> dict[int, _Communicator]:
        comms_list = _Communicator.initAll(list(devices))
        return {comm.device_id(): comm for comm in comms_list}

    def _transfer(
        src_comm: _Communicator, src_stream: Stream, src_data: _AsyncData,
        dst_comm: _Communicator, dst_stream: Stream, dst_dev: int,
    ) -> _AsyncData:
        src_dev = src_data.array.device.id
        if src_dev == dst_dev:
            return _AsyncData(src_data.array, src_data.ready)

        prev_src_stream = get_current_stream(src_dev)
        prev_dst_stream = get_current_stream(dst_dev)
        try:
            with Device(src_dev):
                src_stream.use()
                src_stream.wait_event(src_data.ready)
                src_array = _creation_from_data.ascontiguousarray(
                    src_data.array)

            with Device(dst_dev):
                dst_stream.use()
                dst_buf = _creation_basic.empty(
                    src_array.shape, src_array.dtype)

            dtype, count = _get_nccl_dtype_and_count(src_array)
            nccl.groupStart()

            with Device(src_dev):
                src_comm.send(src_array.data.ptr, count, dtype,
                              dst_comm.rank_id(), src_stream.ptr)

            with Device(dst_dev):
                dst_comm.recv(dst_buf.data.ptr, count, dtype,
                              src_comm.rank_id(), dst_stream.ptr)

                nccl.groupEnd()
                return _AsyncData(dst_buf, dst_stream.record(),
                                  prevent_gc=src_data)
        finally:
            with Device(src_dev):
                prev_src_stream.use()
            with Device(dst_dev):
                prev_dst_stream.use()
else:
    def _create_communicators(
        devices: Iterable[int],
    ) -> dict[int, _Communicator]:
        return {dev: _Communicator() for dev in devices}

    def _transfer(
        src_comm: _Communicator, src_stream: Stream, src_data: _AsyncData,
        dst_comm: _Communicator, dst_stream: Stream, dst_dev: int,
    ) -> _AsyncData:
        src_dev = src_data.array.device.id
        if src_dev == dst_dev:
            return _AsyncData(src_data.array, src_data.ready)

        with Device(dst_dev):
            prev_stream = get_current_stream()
            try:
                dst_stream.use()
                dst_stream.wait_event(src_data.ready)

                dst_array = src_data.array.copy()
                return _AsyncData(
                    dst_array, dst_stream.record(), prevent_gc=src_data.array)
            finally:
                prev_stream.use()
