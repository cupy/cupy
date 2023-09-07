import dataclasses
from typing import Optional

import cupy
from cupyx.distributed import _array


_Shape = tuple[int, ...]
_DeviceMapping = dict[int, tuple[slice, ...]]
_BlockIdx = tuple[tuple[int, int, int], ...]
_InvDeviceMapping = dict[_BlockIdx, set[int]]
_ExecutionPlan = list[tuple[_BlockIdx, _BlockIdx, int]]


@dataclasses.dataclass
class _Blocking:
    i_split: Optional[int]
    j_split: Optional[int]
    k_split: Optional[int]


def _make_inv_mapping(
        shape: _Shape, device_mapping: _DeviceMapping,
    ) -> _InvDeviceMapping:
    inv_mapping = {}

    for dev, slices in device_mapping.items():
        assert len(slices) == len(shape)
        tuples = tuple(s.indices(l) for s, l in zip(slices, shape))
        inv_mapping.setdefault(tuples, set()).add(dev)

    return inv_mapping


def _find_blocking(
        n: int, m: int, p: int,
        inv_mapping_a: _InvDeviceMapping,
        inv_mapping_b: _InvDeviceMapping,
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

    for i_indices, k_indices in inv_mapping_a:
        i_split = update_split(i_indices, n, i_split)
        k_split = update_split(k_indices, m, k_split)

    for k_indices, j_indices in inv_mapping_b:
        k_split = update_split(k_indices, m, k_split)
        j_split = update_split(j_indices, p, j_split)

    return _Blocking(i_split, j_split, k_split)


def _make_execution_plan(
        n: int, m: int, p: int,
        blocking: _Blocking,
        inv_mapping_a: _InvDeviceMapping,
        inv_mapping_b: _InvDeviceMapping
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
                intersection = inv_mapping_a[block_a] & inv_mapping_b[block_b]
                if intersection:
                    dev = intersection.pop()
                    plan.append((block_a, block_b, dev))

    return plan


def matmul(a, b, out=None, **kwargs):
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

    if a.ndim != 2 or b.ndim != 2:
        raise RuntimeError('Only ndim == 2 is supported')
    n, m = a.shape
    m2, p = b.shape
    if m != m2:
        raise ValueError('Shapes are incompatible')

    inv_mapping_a = _make_inv_mapping(a.shape, a._device_mapping)
    inv_mapping_b = _make_inv_mapping(b.shape, b._device_mapping)

    blocking = _find_blocking(n, m, p, inv_mapping_a, inv_mapping_b)
    plan = _make_execution_plan(n, m, p, blocking, inv_mapping_a, inv_mapping_b)

    dev_set = {dev for _, _, dev in plan}
    if len(dev_set) != len(plan):
        raise RuntimeError(
            'Tried to execute multiplications of blocks multiple times on a'
            ' single device')

    chunks = {}
    device_mapping = {}
    dtype = None
    for block_a, block_b, dev in plan:
        chunk_a = a._chunks[dev]
        chunk_b = b._chunks[dev]
        with cupy.cuda.Device(dev):
            stream = cupy.cuda.get_current_stream()
            stream.wait_event(chunk_a.stream.record())
            stream.wait_event(chunk_b.stream.record())
            chunk_ab_data = cupy.matmul(chunk_a.data, chunk_b.data, **kwargs)
            dtype = chunk_ab_data.dtype
            chunks[dev] = _array._ManagedData(chunk_ab_data, stream)
        device_mapping[dev] = (slice(*block_a[0]), slice(*block_b[1]))

    return _array._DistributedArray(
        (n, p), dtype, chunks, device_mapping, _array._MODES['sum'], a._comms)
