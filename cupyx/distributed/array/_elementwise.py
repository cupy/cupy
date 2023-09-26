import typing
from typing import Sequence
from itertools import chain

import cupy
import cupyx.distributed.array as darray
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _data_transfer
from cupyx.distributed.array import _index_arith


def _find_updates(
    args: Sequence['darray.DistributedArray'],
    kwargs: dict[str, 'darray.DistributedArray'],
    dev: int, chunk_i: int,
) -> list['_data_transfer._PartialUpdate']:
    # If there is at most one array with partial updates, we return them
    # and execute the kernel without actually pushing those updates;
    # otherwise we propagate them beforehand.
    # This strategy is slower when many updates overlap with each other.
    # One cause is resharding from an index_map with big overlaps.
    updates: list[_data_transfer._PartialUpdate] = []
    at_most_one_update = True

    for arg in chain(args, kwargs.values()):
        updates_now = arg._chunks_map[dev][chunk_i].updates
        if updates_now:
            if updates:
                at_most_one_update = False
                break
            updates = updates_now

    if at_most_one_update:
        return updates

    for arg in chain(args, kwargs.values()):
        for chunk in chain.from_iterable(arg._chunks_map.values()):
            chunk.apply_updates(arg._mode)
    return []


def _prepare_chunks_data(
    stream: cupy.cuda.Stream,
    args: Sequence['darray.DistributedArray'],
    kwargs: dict[str, 'darray.DistributedArray'],
    dev: int, chunk_i: int,
) -> tuple[list[cupy.ndarray], dict[str, cupy.ndarray]]:
    def access_data(d_array):
        chunk = d_array._chunks_map[dev][chunk_i]
        stream.wait_event(chunk.ready)
        return chunk.data

    new_args = [access_data(arg) for arg in args]
    new_kwargs = {key: access_data(arg) for key, arg in kwargs.items()}

    return new_args, new_kwargs


def _change_all_to_replica_mode(
        args: list['darray.DistributedArray'],
        kwargs: dict[str, 'darray.DistributedArray']) -> None:
    args[:] = [arg._to_replica_mode() for arg in args]
    kwargs.update((k, arg._to_replica_mode()) for k, arg in kwargs.items())


def _execute_kernel(
    kernel,
    args: Sequence['darray.DistributedArray'],
    kwargs: dict[str, 'darray.DistributedArray'],
) -> 'darray.DistributedArray':
    args = list(args)

    # TODO: Skip conversion to the replica mode when mode.func == kernel
    # For example, cupy.add can be done within the sum mode
    _change_all_to_replica_mode(args, kwargs)

    dtype = None
    chunks_map: dict[int, list[_chunk._Chunk]] = {}

    for arg in (args or kwargs.values()):
        index_map = arg.index_map
        break

    for dev, idxs in index_map.items():
        chunks_map[dev] = []
        with cupy.cuda.Device(dev):
            stream = cupy.cuda.get_current_stream()

            for chunk_i, idx in enumerate(idxs):
                # This must be called before _prepare_chunks_data.
                # _find_updates may call _apply_updates, which replaces
                # a placeholder with an actual chunk
                updates = _find_updates(args, kwargs, dev, chunk_i)

                args_data, kwargs_data = _prepare_chunks_data(
                    stream, args, kwargs, dev, chunk_i)

                new_chunk = None
                for data in chain(args_data, kwargs_data.values()):
                    if isinstance(data, _chunk._DataPlaceholder):
                        # A placeholder will be entirely overwritten anyway, so
                        # we just leave it. _find_updates ensures there is
                        # at most one placeholder
                        assert new_chunk is None
                        new_chunk = _chunk._Chunk.create_placeholder(
                            data.shape, data.device, idx)

                if new_chunk is None:
                    # No placeholder
                    new_data = kernel(*args_data, **kwargs_data)

                    dtype = new_data.dtype
                    new_chunk = _chunk._Chunk(
                        new_data, stream.record(), idx,
                        prevent_gc=(args_data, kwargs_data))

                chunks_map[dev].append(new_chunk)

                if not updates:
                    continue

                args_slice = [None] * len(args_data)
                kwargs_slice = {}
                for update, idx in updates:
                    for i, data in enumerate(args_data):
                        if isinstance(data, _chunk._DataPlaceholder):
                            args_slice[i] = update.data
                        else:
                            args_slice[i] = data[idx]
                    for k, data in kwargs_data.items():
                        if isinstance(data, _chunk._DataPlaceholder):
                            kwargs_slice[k] = update.data
                        else:
                            kwargs_slice[k] = data[idx]

                    stream.wait_event(update.ready)
                    new_data = kernel(*args_slice, **kwargs_slice)
                    dtype = new_data.dtype
                    execution_done = stream.record()

                    data_transfer = _data_transfer._AsyncData(
                        new_data, execution_done,
                        prevent_gc=(args_slice, kwargs_slice))
                    new_chunk.add_update(data_transfer, idx)

    for chunk in chain.from_iterable(chunks_map.values()):
        if not isinstance(chunk.data, (cupy.ndarray, _chunk._DataPlaceholder)):
            raise RuntimeError(
                'Kernels returning other than signle array are not supported')

    shape = comms = None
    for arg in (args or kwargs.values()):
        shape = arg.shape
        comms = arg._comms
        break

    assert shape is not None

    return darray.DistributedArray(
        shape, dtype, chunks_map, darray._REPLICA_MODE, comms)


def _execute_peer_access(
    kernel,
    args: Sequence['darray.DistributedArray'],
    kwargs: dict[str, 'darray.DistributedArray'],
) -> 'darray.DistributedArray':
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

    args = list(args)
    for i, arg in enumerate(args):
        args[i] = arg._to_replica_mode()
        for chunk in chain.from_iterable(args[i]._chunks_map.values()):
            chunk.apply_updates(darray._REPLICA_MODE)

    a, b = args

    # TODO: Use numpy.result_type. Does it give the same result?
    if isinstance(kernel, cupy.ufunc):
        op = kernel._ops._guess_routine_from_in_types((a.dtype, b.dtype))
        if op is None:
            raise RuntimeError(
                f'Could not guess the return type of {kernel.name}'
                f' with arguments of type {(a.dtype.type, b.dtype.type)}')
        out_types = op.out_types
    else:
        assert isinstance(kernel, cupy._core._kernel.ElementwiseKernel)
        _, out_types, _ = kernel._decide_params_type(
            (a.dtype.type, b.dtype.type), ())

    if len(out_types) != 1:
        print(out_types)
        raise RuntimeError(
            'Kernels returning other than signle array are not supported')
    dtype = out_types[0]

    shape = a.shape
    comms = a._comms
    chunks_map: dict[int, list[_chunk._Chunk]] = {}

    for a_chunk in chain.from_iterable(a._chunks_map.values()):
        a_dev = a_chunk.data.device.id
        with cupy.cuda.Device(a_dev):
            stream = cupy.cuda.get_current_stream()
            stream.wait_event(a_chunk.ready)

            new_chunk_data = cupy.empty(a_chunk.data.shape, dtype)

            for b_chunk in chain.from_iterable(b._chunks_map.values()):
                stream.wait_event(b_chunk.ready)

                intersection = _index_arith._index_intersection(
                    a_chunk.index, b_chunk.index, shape)
                if intersection is None:
                    continue

                cupy._core._kernel._check_peer_access(b_chunk.data, a_dev)

                a_new_idx = _index_arith._index_for_subindex(
                    a_chunk.index, intersection, shape)
                b_new_idx = _index_arith._index_for_subindex(
                    b_chunk.index, intersection, shape)

                assert kernel.nin == 2
                kernel(typing.cast(cupy.ndarray, a_chunk.data)[a_new_idx],
                       typing.cast(cupy.ndarray, b_chunk.data)[b_new_idx],
                       new_chunk_data[a_new_idx])

            new_chunk = _chunk._Chunk(
                new_chunk_data, stream.record(), a_chunk.index,
                prevent_gc=b._chunks_map)
            chunks_map.setdefault(a_dev, []).append(new_chunk)

    return darray.DistributedArray(
        shape, dtype, chunks_map, darray._REPLICA_MODE, comms)


def _is_peer_access_needed(
    args: Sequence['darray.DistributedArray'],
    kwargs: dict[str, 'darray.DistributedArray'],
) -> bool:
    index_map = None
    for arg in chain(args, kwargs.values()):
        if index_map is None:
            index_map = arg.index_map
        elif arg.index_map != index_map:
            return True

    return False


def _execute(kernel, args: tuple, kwargs: dict):
    for arg in chain(args, kwargs.values()):
        if not isinstance(arg, darray.DistributedArray):
            raise RuntimeError(
                'Mixing a distributed array with a non-distributed one is'
                ' not supported')

    # TODO: check if all distributed
    peer_access = _is_peer_access_needed(args, kwargs)
    if peer_access:
        return _execute_peer_access(kernel, args, kwargs)
    else:
        return _execute_kernel(kernel, args, kwargs)
