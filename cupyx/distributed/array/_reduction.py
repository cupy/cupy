import typing
from typing import Any

from numpy.typing import DTypeLike

import cupy
import cupyx.distributed.array as darray
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array._chunk import _Chunk


def _execute(
    arr: 'darray.DistributedArray', kernel, axis: int, dtype: DTypeLike,
) -> Any:
    overwrites = False
    if kernel.name == 'cupy_max':
        mode = darray._MODES['max']
        if arr._mode is not mode:
            arr = arr._to_replica_mode()
            overwrites = True
    elif kernel.name == 'cupy_min':
        mode = darray._MODES['min']
        if arr._mode is not mode:
            arr = arr._to_replica_mode()
            overwrites = True
    elif kernel.name == 'cupy_sum':
        mode = typing.cast(darray._OpMode, darray._MODES['sum'])
        arr = arr._to_op_mode(mode)
    elif kernel.name == 'cupy_prod':
        mode = typing.cast(darray._OpMode, darray._MODES['prod'])
        arr = arr._to_op_mode(mode)
    else:
        raise RuntimeError(f'Unsupported kernel: {kernel.name}')

    chunks_map = arr._chunks_map

    if overwrites:
        mode = typing.cast(darray._OpMode, mode)
        identity = mode.identity_of(arr.dtype)
        for chunks in chunks_map.values():
            for i in range(len(chunks)):
                if len(chunks[i].updates) == 0:
                    continue
                chunks[i] = chunks[i].copy()
                chunks[i].set_identity_on_overwritten_entries(identity)

    shape = arr.shape[:axis] + arr.shape[axis+1:]
    new_dtype = None
    new_chunks_map: dict[int, list[_Chunk]] = {}

    for dev, chunks in chunks_map.items():
        new_chunks_map[dev] = []
        for chunk in chunks:
            with cupy.cuda.Device(dev):
                execution_stream = cupy.cuda.get_current_stream()
                execution_stream.wait_event(chunk.ready)

                new_index = chunk.index[:axis] + chunk.index[axis+1:]

                if isinstance(chunk.data, _chunk._DataPlaceholder):
                    old_shape = chunk.data.shape
                    new_shape = old_shape[:axis] + old_shape[axis+1:]
                    new_chunk = _Chunk.create_placeholder(
                        new_shape, chunk.data.device, new_index)
                else:
                    # We avoid 0D array because
                    # we expect data[idx] to return a view
                    update_data = cupy.atleast_1d(
                        kernel(chunk.data, axis=axis, dtype=dtype))

                    new_dtype = update_data.dtype
                    new_chunk = _Chunk(
                        update_data, execution_stream.record(), new_index, [],
                        prevent_gc=chunk._prevent_gc)

                new_chunks_map[dev].append(new_chunk)

                if len(chunk.updates) == 0:
                    continue

                for update, update_index in chunk.updates:
                    execution_stream.wait_event(update.ready)
                    new_update_data = cupy.atleast_1d(
                        kernel(update.data, axis=axis, dtype=dtype))
                    new_dtype = new_update_data.dtype

                    data_transfer = _data_transfer._AsyncData(
                        new_update_data, execution_stream.record(),
                        prevent_gc=update.prevent_gc)
                    new_index = update_index[:axis] + update_index[axis+1:]
                    new_chunk.add_update(data_transfer, new_index)

    return darray.DistributedArray(
        shape, new_dtype, new_chunks_map, mode, arr._comms)
