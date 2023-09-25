import dataclasses
import typing
from typing import Any, Callable, Final, Iterable, Optional, TypeVar, Union
from typing_extensions import TypeGuard
from cupyx.distributed.array import _elementwise
from cupyx.distributed.array import _modes


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


@dataclasses.dataclass
class ManagedData:
    data: cupy.ndarray
    ready: Event
    prevent_gc: Any

    def __init__(
        self, data: cupy.ndarray, ready: Event, prevent_gc: Any = None,
    ) -> None:
        self.data = data
        self.ready = ready
        self.prevent_gc = prevent_gc

    def copy(self) -> 'ManagedData':
        with self.data.device:
            stream = get_current_stream()
            stream.wait_event(self.ready)
            update_data = self.data.copy()
            self.ready.record(stream)
            return ManagedData(update_data, stream.record(), self.prevent_gc)


@dataclasses.dataclass
class DataTransfer:
    data: cupy.ndarray
    ready: Event
    prevent_gc: Any = None


# Overwrite in replica mode, apply in op mode
PartialUpdate = tuple[DataTransfer, tuple[slice, ...]]


if nccl.available:
    def create_communicators(
        devices: Iterable[int],
    ) -> dict[int, NcclCommunicator]:
        comms_list = NcclCommunicator.initAll(list(devices))
        return {comm.device_id(): comm for comm in comms_list}


    def transfer(
        src_comm: NcclCommunicator, src_stream: Stream, src_data: ManagedData,
        dst_comm: NcclCommunicator, dst_stream: Stream, dst_dev: int
    ) -> DataTransfer:
        src_dev = src_data.data.device.id
        if src_dev == dst_dev:
            return DataTransfer(src_data.data, src_data.ready)

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
            return DataTransfer(dst_buf, dst_stream.record(),
                                prevent_gc=(src_data, src_array))
else:
    def create_communicators(
        devices: Iterable[int],
    ) -> dict[int, NcclCommunicator]:
        return {dev: NcclCommunicator() for dev in devices}


    def transfer(
        src_comm: NcclCommunicator, src_stream: Stream, src_data: ManagedData,
        dst_comm: NcclCommunicator, dst_stream: Stream, dst_dev: int
    ) -> DataTransfer:
        src_dev = src_data.data.device.id
        if src_dev == dst_dev:
            return DataTransfer(src_data.data, src_data.ready)

        with Device(dst_dev):
            dst_stream.wait_event(src_data.ready)
            with dst_stream:
                dst_data = src_data.data.copy()
            return DataTransfer(dst_data, dst_stream.record(),
                                 prevent_gc=src_data.data)
