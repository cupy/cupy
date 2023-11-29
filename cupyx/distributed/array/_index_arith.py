import typing
from typing import Any, Optional


def _extgcd(a: int, b: int) -> tuple[int, int]:
    # Return (g, x) with g = gcd(a, b), ax + by = g - ax.

    # c - ax - by = 0  ...  (1)
    # d - au - bv = 0  ...  (2)
    c, d = a, b
    x, u = 1, 0
    # y, v = 0, 1

    # Apply Euclid's algorithm to (c, d)
    while d:
        r = c // d
        # (1), (2) = (2), (1) - (2) * r
        c, d = d, c - d * r
        x, u = u, x - u * r
        # y, v = v, y - u * r

    return c, x


def _crt(a1: int, n1: int, a2: int, n2: int) -> Optional[tuple[int, int]]:
    # Return x, n with x == a1 (mod n1), x == a2 (mod n2), n == lcm(n1, n2).
    # Choose the minimum value for x with x >= max(a1, a2). Return None if no
    # solution exists.
    # Reference: https://en.wikipedia.org/wiki/Chinese_remainder_theorem#Existence_(constructive_proof)  # NOQA

    # m1 * n1 + m2 * n2 == g
    g, m1 = _extgcd(n1, n2)

    # x == a1 == a2 (mod g)
    if (a2 - a1) % g != 0:
        return None

    n = n1 * (n2 // g)

    # x = a1 + (a2 - a1) // g * m1 * n1 % n
    x = a1 + (a2 - a1) // g * m1 % (n // n1) * n1

    if x < a2:
        x += ((a2 - x - 1) // n + 1) * n

    return x, n


def _slice_intersection(a: slice, b: slice, length: int) -> Optional[slice]:
    # Return the intersection of slice a, b. None if they are disjoint.
    a_start, a_stop, a_step = a.indices(length)
    b_start, b_stop, b_step = b.indices(length)

    crt_result = _crt(a_start, a_step, b_start, b_step)
    if crt_result is None:
        return None
    c_start, c_step = crt_result

    c_stop = min(a_stop, b_stop)
    if c_start >= c_stop:
        return None

    return slice(c_start, c_stop, c_step)


def _index_for_subslice(a: slice, sub: slice, length: int) -> slice:
    # Return slice c such that array[a][c] == array[sub].
    # sub should be contained in a.
    a_start, a_stop, a_step = a.indices(length)
    sub_start, sub_stop, sub_step = sub.indices(length)

    c_start = (sub_start - a_start) // a_step
    # a_start + a_step * (c_stop - 1) < sub_stop
    c_stop = (sub_stop - a_start - 1) // a_step + 1
    c_step = sub_step // a_step

    return slice(c_start, c_stop, c_step)


def _index_intersection(
    a_idx: tuple[slice, ...], b_idx: tuple[slice, ...],
    shape: tuple[int, ...],
) -> Optional[tuple[slice, ...]]:
    # Return None if a, b are disjoint.
    assert len(a_idx) == len(b_idx)
    result = tuple(_slice_intersection(a, b, length)
                   for a, b, length in zip(a_idx, b_idx, shape))

    if None in result:
        return None
    else:
        return typing.cast(tuple[slice, ...], result)


def _index_for_subindex(
    a_idx: tuple[slice, ...], sub_idx: tuple[slice, ...],
    shape: tuple[int, ...],
) -> tuple[slice, ...]:
    assert len(a_idx) == len(sub_idx)

    return tuple(_index_for_subslice(a, sub, length)
                 for a, sub, length in zip(a_idx, sub_idx, shape))


def _shape_after_indexing(
    outer_shape: tuple[int, ...],
    idx: tuple[slice, ...],
) -> tuple[int, ...]:
    shape = list(outer_shape)
    for i in range(len(idx)):
        start, stop, step = idx[i].indices(shape[i])
        shape[i] = (stop - start - 1) // step + 1
    return tuple(shape)


def _normalize_index(shape: tuple[int, ...], idx: Any) -> tuple[slice, ...]:
    # Convert idx to type tuple[slice, ...] with length == ndim.
    # start, stop, step of each slice are set to a non-None value.
    if not isinstance(idx, tuple):
        idx = (idx,)

    ndim = len(shape)
    if len(idx) > ndim:
        raise IndexError(
            'too many indices for array:'
            f' array is {ndim}-dimensional, but {len(idx)} were indexed')
    idx = idx + (slice(None),) * (ndim - len(idx))

    new_idx = []
    for i in range(ndim):
        if isinstance(idx[i], int):
            if idx[i] >= shape[i]:
                raise IndexError(
                    f'Index {idx[i]} is out of bounds'
                    f' for axis {i} with size {shape[i]}')
            new_idx.append(slice(idx[i], idx[i] + 1, 1))
        elif isinstance(idx[i], slice):
            start, stop, step = idx[i].indices(shape[i])
            if step <= 0:
                raise ValueError('Slice step must be positive.')
            if start == stop:
                raise ValueError(f'The index is empty on axis {i}')
            new_idx.append(slice(start, stop, step))
        else:
            raise ValueError(f'Invalid index on axis {i}')

    return tuple(new_idx)


def _normalize_index_map(
    shape: tuple[int, ...], index_map: dict[int, Any],
) -> dict[int, list[tuple[slice, ...]]]:
    new_index_map: dict[int, list[tuple[slice, ...]]] = {}
    for dev, idxs in index_map.items():
        if not isinstance(idxs, list):
            idxs = [idxs]

        idxs = [_normalize_index(shape, idx) for idx in idxs]
        idxs.sort()
        new_index_map[dev] = idxs

    return new_index_map
