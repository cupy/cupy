import unittest

import itertools
import numpy

import cupy
from cupy import testing


def perm(iterable):
    return list(itertools.permutations(iterable))


@testing.parameterize(
    *testing.product({
        'shape': [(4, 4, 4)],
        'indexes': (
            perm(([1, 0], slice(None))) +
            perm(([1, 0], Ellipsis)) +
            perm(([1, 2], None, slice(None))) +
            perm(([1, 0], 1, slice(None))) +
            perm(([1, 2], slice(0, 2), slice(None))) +
            perm((1, [1, 2], 1)) +
            perm(([[1, -1], [0, 3]], slice(None), slice(None))) +
            perm(([1, 0], [3, 2], slice(None))) +
            perm((slice(0, 3, 2), [1, 2], [1, 0])) +
            perm(([1, 0], [2, 1], [3, 1])) +
            perm(([1, 0], 1, [3, 1])) +
            perm(([1, 2], [[1, 0], [0, 1], [-1, 1]], slice(None))) +
            perm((None, [1, 2], [1, 0]))
        )
    })
)
@testing.gpu
class TestArrayAdvancedIndexingPerm(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_getitem(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return a[self.indexes]


@testing.parameterize(
    {'shape': (2, 3, 4), 'indexes': (None, [1, 0], [0, 2], slice(None))},
    {'shape': (2, 3, 4), 'indexes': (None, [0, 1], None, [2, 1], slice(None))},
    {'shape': (2, 3, 4), 'indexes': numpy.array([1, 0])},
    {'shape': (2, 3, 4), 'indexes': [1, -1]},
)
@testing.gpu
class TestArrayAdvancedIndexingParametrized(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_getitem(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return a[self.indexes]


@testing.parameterize(
    {'shape': (2, 3, 4), 'transpose': (1, 2, 0),
     'indexes': (slice(None), [1, 0])},
    {'shape': (2, 3, 4), 'transpose': (1, 0, 2),
     'indexes': (None, [1, 2], [0, -1])},
)
@testing.gpu
class TestArrayAdvancedIndexingParametrizedTransp(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_getitem(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        if self.transpose:
            a = a.transpose(self.transpose)
        return a[self.indexes]


@testing.parameterize(
    {'shape': (2, 3, 4), 'indexes': (slice(None),)},
    {'shape': (2, 3, 4), 'indexes': (numpy.array([1, 0],))},
    {'shape': (2, 3, 4),
     'indexes': (numpy.random.choice([False, True], (2, 3, 4)),)},
    {'shape': (10,), 'indexes': (numpy.random.choice([False, True], (10,)),)},
)
@testing.gpu
class TestArrayAdvancedIndexingArrayClass(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_getitem(self, xp, dtype):
        indexes = list(self.indexes)
        a = testing.shaped_arange(self.shape, xp, dtype)

        if xp is numpy:
            for i, s in enumerate(indexes):
                if isinstance(s, cupy.ndarray):
                    indexes[i] = s.get()

        return a[tuple(indexes)]


@testing.parameterize(
    {'shape': (), 'indexes': ([1],)},
    {'shape': (2, 3), 'indexes': (slice(None), [1, 2], slice(None))},
)
@testing.gpu
class TestArrayInvalidIndexAdv(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_invalid_adv_getitem(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        a[self.indexes]
