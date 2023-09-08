import dataclasses
import typing
from typing import Optional

import cupy
from cupyx.distributed import _array


_Shape = tuple[int, ...]
_IndexMap = dict[int, list[tuple[slice, ...]]]
_LocalIndexMap = dict[int, list[tuple[slice, slice]]]
_BlockIdx = tuple[tuple[int, int, int], tuple[int, int, int]]
_ChunkLocations = dict[int, int]
_BlockLocationMap = dict[_BlockIdx, _ChunkLocations]
_ExecutionPlan = list[tuple[_BlockIdx, _BlockIdx, int]]
_BatchIdx = tuple[tuple[int, int, int], ...]    # len(batch_idx) == ndim - 2


@dataclasses.dataclass
class _Blocking:
    i_split: Optional[int]
    j_split: Optional[int]
    k_split: Optional[int]


def _convert_to_tuples(
        slices: tuple[slice, ...], shape: tuple[int, ...]
    ) -> tuple[tuple[int, int, int], ...]:
    assert len(slices) == len(shape)
    return tuple(s.indices(l) for s, l in zip(slices, shape))


def _convert_to_slices(tuples: tuple[tuple[int, int, int], ...]
                       ) -> tuple[slice, ...]:
    return tuple(slice(*t) for t in tuples)


def _find_blocking(
        n: int, m: int, p: int,
        location_map_a: _BlockLocationMap,
        location_map_b: _BlockLocationMap,
    ) -> _Blocking:
    undefined = object()
    i_split = j_split = k_split = undefined

    def update_split(indices, length, old_split):
        start, stop, step = indices

        if step != 1:
            raise RuntimeError('Step other than 1 is not supported')

        if (start, stop) == (0, length):
            new_split = None
        elif start == 0:
            new_split = stop
        elif stop == length:
            new_split = start
        else:
            raise RuntimeError(
                'Splitting into more than 2 blocks with respect to one axis is'
                ' not supported')

        if old_split is not undefined and old_split != new_split:
            raise RuntimeError('Inconsistent blocking')

        return new_split

    for i_indices, k_indices in location_map_a.keys():
        i_split = update_split(i_indices, n, i_split)
        k_split = update_split(k_indices, m, k_split)

    for k_indices, j_indices in location_map_b.keys():
        k_split = update_split(k_indices, m, k_split)
        j_split = update_split(j_indices, p, j_split)

    return _Blocking(i_split, j_split, k_split) # type: ignore


def _make_execution_plan(
        n: int, m: int, p: int,
        blocking: _Blocking,
        location_map_a: _BlockLocationMap,
        location_map_b: _BlockLocationMap
    ) -> _ExecutionPlan:
    i_borders = [0, blocking.i_split, n] if blocking.i_split else [0, n]
    j_borders = [0, blocking.j_split, p] if blocking.j_split else [0, p]
    k_borders = [0, blocking.k_split, m] if blocking.k_split else [0, m]

    plan: _ExecutionPlan = []

    for i_range in zip(i_borders, i_borders[1:]):
        for j_range in zip(j_borders, j_borders[1:]):
            for k_range in zip(k_borders, k_borders[1:]):
                block_a = (i_range + (1,), k_range + (1,))
                block_b = (k_range + (1,), j_range + (1,))

                devices_a = set(location_map_a[block_a].keys())
                devices_b = set(location_map_b[block_b].keys())

                intersection = devices_a & devices_b
                if intersection:
                    dev = intersection.pop()
                    plan.append((block_a, block_b, dev))
                else:
                    raise RuntimeError(
                        'There is no device that can perform multiplication'
                        f' between block {block_a} and {block_b}')

    return plan


def _make_batch_idxs(shape: _Shape, index_map: _IndexMap) -> set[_BatchIdx]:
    batch_idxs: set[_BatchIdx] = set()
    for idxs in index_map.values():
        for idx in idxs:
            batch_idxs.add(_convert_to_tuples(idx[:-2], shape[:-2]))
    return batch_idxs


def _make_local_maps(
        batch_idx: _BatchIdx, shape: _Shape, index_map: _IndexMap
    ) -> tuple[_LocalIndexMap, _BlockLocationMap]:
    local_index_map: _LocalIndexMap = {}
    locatoin_map: _BlockLocationMap = {}

    for dev, idxs in index_map.items():
        for chunk_i, idx in enumerate(idxs):
            this_batch_idx = _convert_to_tuples(idx[:-2], shape[:-2])

            if this_batch_idx == batch_idx:
                local_idx = idx[-2:]
                local_index_map.setdefault(dev, []).append(local_idx)

                local_idx = _convert_to_tuples(local_idx, shape[-2:])
                locations = locatoin_map.setdefault(local_idx, {})
                locations[dev] = chunk_i

    return local_index_map, locatoin_map


def matmul(a, b, out=None, **kwargs) -> '_array._DistributedArray':
    if out is not None:
        raise RuntimeError('Argument `out` is not supported')
    for param in ('subok', 'axes', 'axis'):
        if param in kwargs:
            raise RuntimeError(f'Argument `{param}` is not supported')
    if (not isinstance(a, _array._DistributedArray)
            or not isinstance(b, _array._DistributedArray)):
        raise RuntimeError(
            'Mixing a distributed array with a non-distributed array is not'
            ' supported')

    a = a.to_replica_mode()
    b = b.to_replica_mode()


    if a.ndim < 2 or b.ndim < 2:
        raise RuntimeError('ndim < 2 is not supported')
    n, m = a.shape[-2:]
    m2, p = b.shape[-2:]
    if m != m2 or a.shape[:-2] != b.shape[:-2]:
        raise ValueError('Shapes are incompatible')

    batch_idxs = _make_batch_idxs(a.shape, a.index_map)
    if batch_idxs != _make_batch_idxs(b.shape, b.index_map):
        raise RuntimeError('Mismatched batch shapes')

    chunks_map: dict[int, list[_array._Chunk]] = {dev: [] for dev in a.devices}
    dtype = None

    for batch_idx in batch_idxs:
        index_map_a, location_map_a = _make_local_maps(
            batch_idx, a.shape, a.index_map)
        index_map_b, location_map_b = _make_local_maps(
            batch_idx, b.shape, b.index_map)

        blocking = _find_blocking(n, m, p, location_map_a, location_map_b)
        plan = _make_execution_plan(
            n, m, p, blocking, location_map_a, location_map_b)

        index_prefix = _convert_to_slices(batch_idx)
        for block_a, block_b, dev in plan:
            loc_a = location_map_a[block_a]
            loc_b = location_map_b[block_b]
            chunk_a = a._chunks_map[dev][loc_a[dev]]
            chunk_b = b._chunks_map[dev][loc_b[dev]]
            index = index_prefix + (slice(*block_a[0]), slice(*block_b[1]))
            with cupy.cuda.Device(dev):
                stream = cupy.cuda.get_current_stream()
                stream.wait_event(chunk_a.stream.record())
                stream.wait_event(chunk_b.stream.record())
                chunk_ab_data = cupy.matmul(
                    chunk_a.data, chunk_b.data, **kwargs)
                chunk_ab = _array._Chunk(chunk_ab_data, stream, index)
                chunks_map[dev].append(chunk_ab)
                dtype = chunk_ab_data.dtype

    shape = a.shape[:-2] + (n, p)
    return _array._DistributedArray(
        shape, dtype, chunks_map, _array._MODES['sum'], a._comms)
