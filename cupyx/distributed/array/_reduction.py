import typing
from typing import Any

from numpy.typing import DTypeLike

import cupy._manipulation.dims as _manipulation_dims
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array import _modes


def _execute(
    arr: '_array.DistributedArray', kernel, axis: int, dtype: DTypeLike,
) -> Any:
    overwrites = False
    mode_overrides = {
        'cupy_max': _modes.MAX,
        'cupy_min': _modes.MIN,
        'cupy_sum': _modes.SUM,
        'cupy_prod': _modes.PROD,
    }
    if kernel.name not in mode_overrides:

        raise RuntimeError(f'Unsupported kernel: {kernel.name}')
    mode = mode_overrides[kernel.name]
    if mode in (_modes.MAX, _modes.MIN):
        if arr._mode is not mode:
            arr = arr._to_op_mode(_modes.REPLICA)
            overwrites = True
    else:
        arr = arr._to_op_mode(mode)

    chunks_map = arr._chunks_map

    if overwrites:
        mode = typing.cast(_modes._OpMode, mode)
        identity = mode.identity_of(arr.dtype)
        for chunks in chunks_map.values():
            for i in range(len(chunks)):
                if len(chunks[i].updates) == 0:
                    continue
                chunks[i] = chunks[i].copy()
                chunks[i].set_identity_on_overwritten_entries(identity)

    shape = arr.shape[:axis] + arr.shape[axis+1:]
    out_dtype = None
    out_chunks_map: dict[int, list[_chunk._Chunk]] = {}

    for dev, chunks in chunks_map.items():
        out_chunks_map[dev] = []
        for chunk in chunks:
            with chunk.on_ready() as stream:
                out_index = chunk.index[:axis] + chunk.index[axis+1:]

                if isinstance(chunk.array, _chunk._ArrayPlaceholder):
                    old_shape = chunk.array.shape
                    out_shape = old_shape[:axis] + old_shape[axis+1:]
                    out_chunk = _chunk._Chunk.create_placeholder(
                        out_shape, chunk.array.device, out_index)
                else:
                    # We avoid 0D array because
                    # we expect data[idx] to return a view
                    out_array = _manipulation_dims.atleast_1d(
                        kernel(chunk.array, axis=axis, dtype=dtype))

                    out_dtype = out_array.dtype
                    out_chunk = _chunk._Chunk(
                        out_array, stream.record(), out_index,
                        prevent_gc=chunk.prevent_gc)

                out_chunks_map[dev].append(out_chunk)

                if len(chunk.updates) == 0:
                    continue

                for update, update_index in chunk.updates:
                    stream.wait_event(update.ready)
                    out_update_array = _manipulation_dims.atleast_1d(
                        kernel(update.array, axis=axis, dtype=dtype))
                    out_dtype = out_update_array.dtype

                    out_update = _data_transfer._AsyncData(
                        out_update_array, stream.record(),
                        prevent_gc=update.prevent_gc)
                    out_index = update_index[:axis] + update_index[axis+1:]
                    out_chunk.add_update(out_update, out_index)

    return _array.DistributedArray(
        shape, out_dtype, out_chunks_map, mode, arr._comms)
