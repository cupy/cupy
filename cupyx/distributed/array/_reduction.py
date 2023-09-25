from typing import Any

from numpy.typing import DTypeLike

import cupyx.distributed.array as darray
from cupyx.distributed.array._modes import *
from cupyx.distributed.array._data_transfer import *


def execute(
    arr: 'darray._DistributedArray', kernel, axis: int, dtype: DTypeLike
) -> Any:
    overwrites = False
    if kernel.name == 'cupy_max':
        mode = MODES['max']
        if arr._mode is mode:
            chunks_map = arr._chunks_map
        else:
            chunks_map = arr._replica_mode_chunks_map()
            overwrites = True
    elif kernel.name == 'cupy_min':
        mode = MODES['min']
        if arr._mode is mode:
            chunks_map = arr._chunks_map
        else:
            chunks_map = arr._replica_mode_chunks_map()
            overwrites = True
    elif kernel.name == 'cupy_sum':
        mode = typing.cast(OpMode, MODES['sum'])
        chunks_map = arr._op_mode_chunks_map(mode)
    elif kernel.name == 'cupy_prod':
        mode = typing.cast(OpMode, MODES['prod'])
        chunks_map = arr._op_mode_chunks_map(mode)
    else:
        raise RuntimeError(f'Unsupported kernel: {kernel.name}')

    if overwrites:
        mode = typing.cast(OpMode, mode)
        identity = mode.identity_of(arr.dtype)
        for chunks in chunks_map.values():
            for i in range(len(chunks)):
                if len(chunks[i].updates) == 0:
                    continue
                chunks[i] = chunks[i].copy()
                arr._set_identity_on_ignored_entries(identity, chunks[i])

    shape = arr.shape[:axis] + arr.shape[axis+1:]
    new_dtype = None
    new_chunks_map: dict[int, list[Chunk]] = {}

    for dev, chunks in chunks_map.items():
        new_chunks_map[dev] = []
        for chunk in chunks:
            with Device(dev):
                execution_stream = get_current_stream()
                execution_stream.wait_event(chunk.ready)

                new_index = chunk.index[:axis] + chunk.index[axis+1:]

                if isinstance(chunk.data, DataPlaceholder):
                    old_shape = chunk.data.shape
                    new_shape = old_shape[:axis] + old_shape[axis+1:]
                    new_chunk = Chunk(
                        DataPlaceholder(new_shape, chunk.data.device),
                        chunk.ready, new_index, [],
                        prevent_gc=chunk.prevent_gc)
                else:
                    update_data = cupy.atleast_1d(
                        kernel(chunk.data, axis=axis, dtype=dtype))

                    new_dtype = update_data.dtype
                    new_chunk = Chunk(
                        update_data, execution_stream.record(), new_index, [],
                        prevent_gc=chunk.prevent_gc)

                new_chunks_map[dev].append(new_chunk)

                if len(chunk.updates) == 0:
                    continue

                for update, update_index in chunk.updates:
                    execution_stream.wait_event(update.ready)
                    new_update_data = cupy.atleast_1d(
                        kernel(update.data, axis=axis, dtype=dtype))
                    new_dtype = new_update_data.dtype

                    data_transfer = DataTransfer(
                        new_update_data, execution_stream.record(),
                        prevent_gc=update.prevent_gc)
                    new_index = update_index[:axis] + update_index[axis+1:]
                    new_chunk.updates.append((data_transfer, new_index))

    return darray._DistributedArray(
        shape, new_dtype, new_chunks_map, mode, arr._comms)
