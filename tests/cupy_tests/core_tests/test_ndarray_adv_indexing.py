import unittest

import numpy

import cupy
from cupy import testing


@testing.parameterize(
    {'shape': (2, 3, 4), 'indexes': (slice(None), [1, 0])},
    {'shape': (2, 3, 4), 'indexes': (slice(None), [1, 0])},
    {'shape': (2, 3, 4), 'indexes': ([1, -1], slice(None))},
    {'shape': (2, 3, 4), 'indexes': (Ellipsis, [1, 0])},
    {'shape': (2, 3, 4), 'indexes': ([1, -1], Ellipsis)},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), slice(None), [[1, -1], [0, 3]])},
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


@testing.parameterize(
    {'shape': (2, 3, 4), 'indexes': ([1, 0], [2, 1])},
)
@testing.gpu
class TestArrayAdvancedIndexingNotSupported(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_not_supported_adv_indexing(self, dtype):
        a = testing.shaped_arange(self.shape, cupy, dtype)
        with self.assertRaises(NotImplementedError):
            a[self.indexes]
