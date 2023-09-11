import dataclasses
import typing
from typing import Optional

import cupy
from cupyx.distributed import _array


_Shape = tuple[int, ...]
_IndexMap = dict[int, list[tuple[slice, ...]]]
_SliceIndices = tuple[int, int, int]
_BlockIdx = tuple[_SliceIndices, _SliceIndices]
_ChunkLocations = dict[int, int]
_BlockLocationMap = dict[_BlockIdx, _ChunkLocations]
_ExecutionPlan = list[tuple[_BlockIdx, _BlockIdx, int]]
_BatchIdx = tuple[_SliceIndices, ...]    # len(batch_idx) == ndim - 2


@dataclasses.dataclass
class _Blocking:
    i_partitions: list[int]
    j_partitions: list[int]
    k_partitions: list[int]


def _convert_to_tuples(
        slices: tuple[slice, ...], shape: tuple[int, ...],
    ) -> tuple[_SliceIndices, ...]:
    assert len(slices) == len(shape)
    return tuple(s.indices(l) for s, l in zip(slices, shape))


def _convert_to_slices(
        tuples: tuple[_SliceIndices, ...]) -> tuple[slice, ...]:
    return tuple(slice(*t) for t in tuples)


def _find_blocking(
        n: int, m: int, p: int,
        location_map_a: _BlockLocationMap,
        location_map_b: _BlockLocationMap,
    ) -> _Blocking:

    i_partitions: list[int] = []
    j_partitions: list[int] = []
    k_partitions: list[int] = []

    def add_partitions(indices, partitions):
        start, stop, step = indices

        if step != 1:
            raise RuntimeError('Step other than 1 is not supported')

        partitions.append(start)
        partitions.append(stop)


    for i_indices, k_indices in location_map_a.keys():
        add_partitions(i_indices, i_partitions)
        add_partitions(k_indices, k_partitions)

    for k_indices, j_indices in location_map_b.keys():
        add_partitions(k_indices, k_partitions)
        add_partitions(j_indices, j_partitions)

    def to_unique_sorted(partitions):
        if len(partitions) == 0:
            raise RuntimeError('An array has no chunk')

        partitions.sort()

        res = [partitions[0]]
        for x, y in zip(partitions, partitions[1:]):
            if x != y:
                res.append(y)

        return res

    i_partitions = to_unique_sorted(i_partitions)
    j_partitions = to_unique_sorted(j_partitions)
    k_partitions = to_unique_sorted(k_partitions)

    def validate_indices(indices, partitions):
        start, stop, _ = indices
        if partitions.index(start) + 1 != partitions.index(stop):
            raise RuntimeError('Inconsistent index mapping')

    for i_indices, k_indices in location_map_a.keys():
        validate_indices(i_indices, i_partitions)
        validate_indices(k_indices, k_partitions)

    for k_indices, j_indices in location_map_b.keys():
        validate_indices(k_indices, k_partitions)
        validate_indices(j_indices, j_partitions)

    return _Blocking(i_partitions, j_partitions, k_partitions)


def _make_execution_plan(
        n: int, m: int, p: int,
        blocking: _Blocking,
        location_map_a: _BlockLocationMap,
        location_map_b: _BlockLocationMap,
    ) -> _ExecutionPlan:

    i_partitions = blocking.i_partitions
    j_partitions = blocking.j_partitions
    k_partitions = blocking.k_partitions

    plan: _ExecutionPlan = []

    for i_range in zip(i_partitions, i_partitions[1:]):
        for j_range in zip(j_partitions, j_partitions[1:]):
            for k_range in zip(k_partitions, k_partitions[1:]):
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
        batch_idx: _BatchIdx, shape: _Shape, index_map: _IndexMap,
    ) -> _BlockLocationMap:
    block_locatoin_map: _BlockLocationMap = {}

    for dev, idxs in index_map.items():
        for chunk_i, idx in enumerate(idxs):
            idx_tuples = _convert_to_tuples(idx, shape)
            this_batch_idx, block_idx = idx_tuples[:-2], idx_tuples[-2:]
            block_idx = typing.cast(_BlockIdx, block_idx)

            if this_batch_idx == batch_idx:
                locations = block_locatoin_map.setdefault(block_idx, {})
                locations[dev] = chunk_i

    return block_locatoin_map


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

    one_prepended = one_appended = False
    if a.ndim == 1:
        one_prepended = True
        a = a._prepend_one_to_shape()
    if b.ndim == 1:
        one_appended = True
        b = b._append_one_to_shape()

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
        location_map_a = _make_local_maps(batch_idx, a.shape, a.index_map)
        location_map_b = _make_local_maps(batch_idx, b.shape, b.index_map)

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
    res = _array._DistributedArray(
        shape, dtype, chunks_map, _array._MODES['sum'], a._comms)

    if one_prepended:
        res = res._pop_front_from_shape()
    if one_appended:
        res = res._pop_from_shape()

    return res


def make_2d_index_map(
    i_partitions: list[int],
    j_partitions: list[int],
    devices: list[list[set[int]]],
) -> dict[int, list[tuple[slice, ...]]]:
    index_map: dict[int, list[tuple[slice, ...]]] = {}
    for i in range(len(i_partitions) - 1):
        for j in range(len(j_partitions) - 1):
            i_start = i_partitions[i]
            i_stop = i_partitions[i+1]
            j_start = j_partitions[j]
            j_stop = j_partitions[j+1]

            idx = (slice(i_start, i_stop), slice(j_start, j_stop))

            for dev in devices[i][j]:
                index_map.setdefault(dev, []).append(idx)

    return index_map
