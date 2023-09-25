import typing
from typing import Any, Optional, Sequence, Union

import itertools

import cupy
import cupyx.distributed.array as darray
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array import _index_arith
from cupyx.distributed.array import _util
from cupyx.distributed.array import _modes


def _prepare_updates(
    args: Sequence['darray._DistributedArray'],
    kwargs: dict[str, 'darray._DistributedArray'],
    dev: int, chunk_i: int,
) -> tuple[Optional[Union[int, str]], list['_data_transfer.PartialUpdate']]:
    index = None
    updates: list[_data_transfer.PartialUpdate] = []
    at_most_one_update = True

    for i, arg in itertools.chain(enumerate(args), kwargs.items()):
        updates_now = arg._chunks_map[dev][chunk_i].updates
        if updates_now:
            if updates:
                at_most_one_update = False
                break
            index = i
            updates = updates_now

    # If there is at most one array with partial updates, we return them
    # and apply the element-wise kernel without actually propagating
    # those updates. Otherwise we propagate them beforehand.
    # TODO check if it really gives any speedup
    if at_most_one_update:
        return index, updates

    # TODO leave one chunk with updates
    for arg in itertools.chain(args, kwargs.values()):
        arg._apply_updates_all_chunks()
    return None, []


def _access_chunks_data(
    stream: cupy.cuda.Stream,
    args: Sequence['darray._DistributedArray'],
    kwargs: dict[str, 'darray._DistributedArray'],
    dev: int, chunk_i: int, idx: tuple[slice, ...],
) -> tuple[list[cupy.ndarray],
           dict[str, cupy.ndarray]]:
    def access_datas(d_array):
        chunk = d_array._chunks_map[dev][chunk_i]
        stream.wait_event(chunk.ready)
        return chunk.data

    new_args = [access_datas(arg) for arg in args]
    new_kwargs = {key: access_datas(arg) for key, arg in kwargs.items()}

    return new_args, new_kwargs


def _change_all_to_replica_mode(
        args: list['darray._DistributedArray'],
        kwargs: dict[str, 'darray._DistributedArray']) -> None:
    args[:] = [arg.to_replica_mode() for arg in args]
    kwargs.update((k, arg.to_replica_mode()) for k, arg in kwargs.items())


def execute_kernel(
    kernel,
    args: tuple['darray._DistributedArray'],
    kwargs: dict[str, 'darray._DistributedArray'],
) -> 'darray._DistributedArray':
    args = list(args)
    _change_all_to_replica_mode(args, kwargs)

    dtype = None
    chunks_map: dict[int, list[_chunk.Chunk]] = {}

    for arg in (args or kwargs.values()):
        index_map = arg.index_map
        break

    for dev, idxs in index_map.items():
        chunks_map[dev] = []
        with cupy.cuda.Device(dev):
            stream = cupy.cuda.get_current_stream()

            for chunk_i, idx in enumerate(idxs):
                # This must be called BEFORE _access_chunks_data
                # _prepare_updates may call _apply_updates
                # which replaces a placeholder with an actual chunk
                incoming_index, update_map = _prepare_updates(
                    args, kwargs, dev, chunk_i)

                args_data, kwargs_data = _access_chunks_data(
                    stream, args, kwargs, dev, chunk_i, idx)

                new_chunk = None
                for data in itertools.chain(args_data, kwargs_data.values()):
                    if isinstance(data, _chunk.DataPlaceholder):
                        new_chunk = _chunk.Chunk(data, cupy.cuda.Event(), idx)

                if new_chunk is None:
                    print('execution', args_data)
                    new_data = kernel(*args_data, **kwargs_data)

                    dtype = new_data.dtype
                    new_chunk = _chunk.Chunk(
                        new_data, stream.record(),
                        index_map[dev][chunk_i],
                        prevent_gc=(args_data, kwargs_data))

                chunks_map[dev].append(new_chunk)

                if incoming_index is None:
                    continue

                args_slice = [None] * len(args_data)
                kwargs_slice = {}
                for update, idx in update_map:
                    for i, data in enumerate(args_data):
                        if isinstance(data, _chunk.DataPlaceholder):
                            args_slice[i] = update.data
                        else:
                            args_slice[i] = data[idx]
                    for k, data in kwargs_data.items():
                        if isinstance(data, _chunk.DataPlaceholder):
                            kwargs_slice[k] = update.data
                        else:
                            kwargs_slice[k] = data[idx]

                    stream.wait_event(update.ready)
                    new_data = kernel(*args_slice, **kwargs_slice)
                    dtype = new_data.dtype
                    execution_done = stream.record()

                    data_transfer = _data_transfer.DataTransfer(
                        new_data, execution_done,
                        prevent_gc=(args_slice, kwargs_slice))
                    new_chunk.updates.append((data_transfer, idx))

    for dev, chunk in _util.all_chunks(chunks_map):
        if not isinstance(chunk.data, cupy.ndarray) and not isinstance(chunk.data, _chunk.DataPlaceholder):
            raise RuntimeError(
                'Kernels returning other than signle array are not'
                ' supported')

    shape = comms = None
    for arg in (args or kwargs.values()):
        shape = arg.shape
        comms = arg._comms

    return darray._DistributedArray(
        shape, dtype, chunks_map, _modes.REPLICA_MODE, comms)


def execute_peer_access(
    kernel,
    args: tuple['darray._DistributedArray'],
    kwargs: dict[str, 'darray._DistributedArray'],
) -> 'darray._DistributedArray':
    """Arguments must be in the replica mode."""
    assert len(args) >= 2   # if len == 1, peer access should be unnecessary
    if len(args) > 2:
        raise RuntimeError(
            'Element-wise operation over more than two distributed arrays'
            ' is not supported unless they share the same index_map.')
    if kwargs:
        raise RuntimeError(
            'Keyword argument is not supported'
            ' unless arguments share the same index_map.')

    a, b = args
    a, b = a.to_replica_mode(), b.to_replica_mode()

    # TODO: use numpy.result_type and compare
    if isinstance(kernel, cupy.ufunc):
        op = kernel._ops._guess_routine_from_in_types((a.dtype, b.dtype))
        if op is None:
            raise RuntimeError(
                f'Could not guess the return type of {kernel.name}'
                f' with arguments of type {(a.dtype.type, b.dtype.type)}')
        out_dtypes = op.out_types
    else:
        assert isinstance(kernel, cupy._core._kernel.ElementwiseKernel)
        out_dtypes = kernel._decide_params_type(
            (a.dtype.type, b.dtype.type), ([],))

    if len(out_dtypes) != 1:
        raise RuntimeError(
            'Kernels returning other than signle array are not'
            ' supported')
    dtype = out_dtypes[0]

    shape = a.shape
    comms = a._comms
    chunks_map: dict[int, list[_chunk.Chunk]] = {}

    for a_dev, a_chunk in _util.all_chunks(a._chunks_map):
        assert isinstance(a_chunk.data, cupy.ndarray)

        with cupy.cuda.Device(a_dev):
            stream = cupy.cuda.get_current_stream()
            stream.wait_event(a_chunk.ready)

            new_chunk_data = cupy.empty(a_chunk.data.shape, dtype)

            for b_dev, b_chunk in _util.all_chunks(b._chunks_map):
                assert isinstance(b_chunk.data, cupy.ndarray)

                intersection = _index_arith.index_intersection(
                    a_chunk.index, b_chunk.index, shape)
                if intersection is None:
                    continue

                cupy._core._kernel._check_peer_access(b_chunk.data, a_dev)

                a_new_idx = _index_arith.index_for_subindex(
                    a_chunk.index, intersection, shape)
                b_new_idx = _index_arith.index_for_subindex(
                    b_chunk.index, intersection, shape)

                # TODO check kernel.nin
                stream.wait_event(b_chunk.ready)
                kernel(a_chunk.data[a_new_idx],
                       b_chunk.data[b_new_idx],
                       new_chunk_data[a_new_idx])

            new_chunk = _chunk.Chunk(
                new_chunk_data, stream.record(), a_chunk.index,
                updates=[], prevent_gc=b._chunks_map)
            chunks_map.setdefault(a_dev, []).append(new_chunk)

    return darray._DistributedArray(
        shape, dtype, chunks_map, _modes.REPLICA_MODE, comms)


def _is_peer_access_needed(
    args: tuple['darray._DistributedArray'],
    kwargs: dict[str, 'darray._DistributedArray'],
) -> bool:
    index_map = None
    for arg in itertools.chain(args, kwargs.values()):
        if index_map is None:
            index_map = arg.index_map
        elif arg.index_map != index_map:
            return True

    return False


def execute(kernel, args, kwargs):
    for arg in itertools.chain(args, kwargs.values()):
        if not isinstance(arg, darray._DistributedArray):
            raise RuntimeError(
                'Mixing a distributed array with a non-distributed one is'
                ' not supported')

    # TODO: check if all distributed
    peer_access = _is_peer_access_needed(args, kwargs)
    if peer_access:
        return execute_peer_access(kernel, args, kwargs)
    else:
        return execute_kernel(kernel, args, kwargs)
