import random
import math

from cupyx.distributed.array import _index_arith


def test_extgcd():
    iteration = 300
    max_value = 100

    for _ in range(iteration):
        a = random.randint(1, max_value)
        b = random.randint(1, max_value)
        g, x = _index_arith._extgcd(a, b)
        assert g == math.gcd(a, b)
        assert (g - a * x) % b == 0


def test_slice_intersection():
    iteration = 300
    max_value = 100

    for _ in range(iteration):
        a_start = random.randint(0, max_value - 1)
        b_start = random.randint(0, max_value - 1)
        a_stop = random.randint(a_start + 1, max_value)
        b_stop = random.randint(b_start + 1, max_value)
        a_step = random.randint(1, max_value // 3)
        b_step = random.randint(1, max_value // 3)
        a = slice(a_start, a_stop, a_step)
        b = slice(b_start, b_stop, b_step)

        def all_indices(s0: slice, s1: slice = slice(None)) -> set[int]:
            """Return all indices for the elements of array[s0][s1]."""
            all_indices = list(range(max_value))
            return set(all_indices[s0][s1])

        c = _index_arith._slice_intersection(a, b, max_value)
        if c is None:
            assert not (all_indices(a) & all_indices(b))
        else:
            assert all_indices(c) == all_indices(a) & all_indices(b)
            p = _index_arith._index_for_subslice(a, c, max_value)
            assert all_indices(c) == all_indices(a, p)
