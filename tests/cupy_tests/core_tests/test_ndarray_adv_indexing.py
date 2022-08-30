import itertools
import numpy
import pytest

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
            perm((None, [1, 2], [1, 0])) +
            perm((numpy.array(0), numpy.array(-1))) +
            perm((numpy.array(0), None)) +
            perm((1, numpy.array(2), slice(None)))
        )
    })
)
@testing.gpu
class TestArrayAdvancedIndexingGetitemPerm:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_getitem(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return a[self.indexes]


@testing.parameterize(
    {'shape': (2, 3, 4), 'indexes': numpy.array(-1)},
    {'shape': (2, 3, 4), 'indexes': (None, [1, 0], [0, 2], slice(None))},
    {'shape': (2, 3, 4), 'indexes': (None, [0, 1], None, [2, 1], slice(None))},
    {'shape': (2, 3, 4), 'indexes': numpy.array([1, 0])},
    {'shape': (2, 3, 4), 'indexes': [1]},
    {'shape': (2, 3, 4), 'indexes': [1, 1]},
    {'shape': (2, 3, 4), 'indexes': [1, -1]},
    {'shape': (2, 3, 4), 'indexes': ([0, 1], slice(None), [[2, 1], [3, 1]])},
    # mask
    {'shape': (10,), 'indexes': (numpy.random.choice([False, True], (10,)),)},
    {'shape': (2, 3, 4), 'indexes': (1, numpy.array([True, False, True]))},
    {'shape': (2, 3, 4), 'indexes': (numpy.array([True, False]), 1)},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), 2, numpy.array([True, False, True, False]))},
    {'shape': (2, 3, 4), 'indexes': (slice(None), 2, False)},
    {'shape': (2, 3, 4),
     'indexes': (numpy.random.choice([False, True], (2, 3, 4)),)},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), numpy.array([True, False, True]))},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), slice(None),
                 numpy.array([True, False, False, True]))},
    {'shape': (2, 3, 4),
     'indexes': (1, 2,
                 numpy.array([True, False, False, True]))},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), numpy.random.choice([False, True], (3, 4)))},
    {'shape': (2, 3, 4),
     'indexes': numpy.random.choice([False, True], (2, 3))},
    {'shape': (2, 3, 4),
     'indexes': (1, None, numpy.array([True, False, True]))},
    # empty arrays
    {'shape': (2, 3, 4), 'indexes': []},
    {'shape': (2, 3, 4), 'indexes': numpy.array([], dtype=numpy.int32)},
    {'shape': (2, 3, 4), 'indexes': numpy.array([[]], dtype=numpy.int32)},
    {'shape': (2, 3, 4), 'indexes': (slice(None), [])},
    {'shape': (2, 3, 4), 'indexes': ([], [])},
    {'shape': (2, 3, 4), 'indexes': ([[]],)},
    {'shape': (2, 3, 4), 'indexes': numpy.array([], dtype=numpy.bool_)},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), numpy.array([], dtype=numpy.bool_))},
    {'shape': (2, 3, 4), 'indexes': numpy.array([[], []], dtype=numpy.bool_)},
    {'shape': (2, 3, 4), 'indexes': numpy.empty((0, 0, 4), bool)},
    # multiple masks
    {'shape': (2, 3, 4), 'indexes': (True, [True, False])},
    {'shape': (2, 3, 4), 'indexes': (False, [True, False])},
    {'shape': (2, 3, 4), 'indexes': (True, [[1]], slice(1, 2))},
    {'shape': (2, 3, 4), 'indexes': (False, [[1]], slice(1, 2))},
    {'shape': (2, 3, 4), 'indexes': (True, [[1]], slice(1, 2), True)},
    {'shape': (2, 3, 4), 'indexes': (True, [[1]], slice(1, 2), False)},
    {'shape': (2, 3, 4),
     'indexes': (Ellipsis, [[1, 1, -3], [0, 2, 2]], [True, False, True, True])
     },
    {'shape': (2, 3, 4),
     'indexes': (numpy.empty((0, 3), bool), numpy.empty(0, bool))},
    # zero-dim and zero-sized arrays
    {'shape': (), 'indexes': Ellipsis},
    {'shape': (), 'indexes': ()},
    {'shape': (), 'indexes': None},
    {'shape': (), 'indexes': True},
    {'shape': (), 'indexes': (True,)},
    {'shape': (), 'indexes': (False, True, True)},
    {'shape': (), 'indexes': numpy.ones((), dtype=numpy.bool_)},
    {'shape': (), 'indexes': numpy.zeros((), dtype=numpy.bool_)},
    {'shape': (0,), 'indexes': None},
    {'shape': (0,), 'indexes': ()},
    {'shape': (2, 0), 'indexes': ([1],)},
    {'shape': (0, 3), 'indexes': (slice(None), [1])},
    {'shape': (0,), 'indexes': True},
    {'shape': (0,), 'indexes': (True,)},
    {'shape': (0,), 'indexes': (False, True, True)},
    {'shape': (0,), 'indexes': numpy.ones((), dtype=numpy.bool_)},
    {'shape': (0,), 'indexes': numpy.zeros((), dtype=numpy.bool_)},
    # ellipsis
    {'shape': (2, 3, 4), 'indexes': (1, Ellipsis, 2)},
    # issue #1512
    {'shape': (2, 3, 4), 'indexes': (Ellipsis, numpy.array(False))},
    {'shape': (2, 3, 4), 'indexes': (Ellipsis, numpy.ones((3, 4), bool))},
    # issue #4799
    {'shape': (3, 4, 5),
     'indexes': (slice(None), [0, 1], Ellipsis, [0, 1])},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), [1, 0], Ellipsis, numpy.ones((5, 2), int))},
    _ids=False,  # Do not generate ids from randomly generated params
)
@testing.gpu
class TestArrayAdvancedIndexingGetitemParametrized:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_getitem(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return a[self.indexes]


@testing.parameterize(
    # empty arrays (list indexes)
    {'shape': (2, 3, 4), 'indexes': [[]]},
    {'shape': (2, 3, 4), 'indexes': [[[]]]},
    {'shape': (2, 3, 4), 'indexes': [[[[]]]]},
    {'shape': (2, 3, 4, 5), 'indexes': [[[[]]]]},
    {'shape': (2, 3, 4, 5), 'indexes': [[[[[]]]]]},
    # list indexes
    {'shape': (2, 3, 4), 'indexes': [[1]]},
    {'shape': (2, 3, 4), 'indexes': [[1, 1]]},
    {'shape': (2, 3, 4), 'indexes': [[1], [1]]},
)
@testing.with_requires('numpy>=1.23')
class TestArrayAdvancedIndexingGetitemParametrized2:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_getitem(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        return a[self.indexes]


@testing.parameterize(
    # list indexes
    {'shape': (2, 3, 4), 'indexes': [[1, 1], 1]},
    {'shape': (2, 3, 4), 'indexes': [[1], slice(1, 2)]},
    {'shape': (2, 3, 4), 'indexes': [[[1]], slice(1, 2)]},
)
@testing.with_requires('numpy>=1.23')
class TestArrayAdvancedIndexingGetitemParametrizedIndexError:

    @testing.for_all_dtypes()
    def test_adv_getitem(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp, dtype)
            with pytest.raises(IndexError):
                with testing.assert_warns(numpy.VisibleDeprecationWarning):
                    a[self.indexes]


@testing.parameterize(
    {'shape': (2, 3, 4), 'transpose': (1, 2, 0),
     'indexes': (slice(None), [1, 0])},
    {'shape': (2, 3, 4), 'transpose': (1, 0, 2),
     'indexes': (None, [1, 2], [0, -1])},
)
@testing.gpu
class TestArrayAdvancedIndexingGetitemParametrizedTransp:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_getitem(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        if self.transpose:
            a = a.transpose(self.transpose)
        return a[self.indexes]


@testing.gpu
class TestArrayAdvancedIndexingGetitemCupyIndices:

    shape = (2, 3, 4)

    def test_adv_getitem_cupy_indices1(self):
        a = cupy.zeros(self.shape)
        index = cupy.array([1, 0])
        original_index = index.copy()
        b = a[index]
        b_cpu = a.get()[index.get()]
        testing.assert_array_equal(b, b_cpu)
        testing.assert_array_equal(original_index, index)

    def test_adv_getitem_cupy_indices2(self):
        a = cupy.zeros(self.shape)
        index = cupy.array([1, 0])
        original_index = index.copy()
        b = a[(slice(None), index)]
        b_cpu = a.get()[(slice(None), index.get())]
        testing.assert_array_equal(b, b_cpu)
        testing.assert_array_equal(original_index, index)

    def test_adv_getitem_cupy_indices3(self):
        a = cupy.zeros(self.shape)
        index = cupy.array([True, False])
        original_index = index.copy()
        b = a[index]
        b_cpu = a.get()[index.get()]
        testing.assert_array_equal(b, b_cpu)
        testing.assert_array_equal(original_index, index)

    def test_adv_getitem_cupy_indices4(self):
        a = cupy.zeros(self.shape)
        index = cupy.array([4, -5])
        original_index = index.copy()
        b = a[index]
        b_cpu = a.get()[index.get() % self.shape[1]]
        testing.assert_array_equal(b, b_cpu)
        testing.assert_array_equal(original_index, index)

    def test_adv_getitem_cupy_indices5(self):
        a = cupy.zeros(self.shape)
        index = cupy.array([4, -5])
        original_index = index.copy()
        b = a[[1, 0], index]
        b_cpu = a.get()[[1, 0], index.get() % self.shape[1]]
        testing.assert_array_equal(b, b_cpu)
        testing.assert_array_equal(original_index, index)


@testing.parameterize(
    {'shape': (2**3 + 1, 2**4), 'indexes': (
        numpy.array([2**3], dtype=numpy.int8),
        numpy.array([1], dtype=numpy.int8))},
    {'shape': (2**4 + 1, 2**4), 'indexes': (
        numpy.array([2**4], dtype=numpy.uint8),
        numpy.array([1], dtype=numpy.uint8))},
    {'shape': (2**7 + 1, 2**8), 'indexes': (
        numpy.array([2**7], dtype=numpy.int16),
        numpy.array([1], dtype=numpy.int16))},
    {'shape': (2**8 + 1, 2**8), 'indexes': (
        numpy.array([2**8], dtype=numpy.uint16),
        numpy.array([1], dtype=numpy.uint16))},
    {'shape': (2**7 + 1, 2**8), 'indexes': (
        numpy.array([2**7], dtype=numpy.int16),
        numpy.array([1], dtype=numpy.int32))},
    {'shape': (2**7 + 1, 2**8), 'indexes': (
        numpy.array([2**7], dtype=numpy.int16),
        numpy.array([1], dtype=numpy.int8))},
    # Three-dimensional case
    {'shape': (2**3 + 1, 3, 2**4), 'indexes': (
        numpy.array([2**3], dtype=numpy.int8),
        slice(None),
        numpy.array([1], dtype=numpy.int8))},
)
@testing.gpu
class TestArrayAdvancedIndexingOverflow:

    def test_getitem(self):
        a = cupy.arange(numpy.prod(self.shape)).reshape(self.shape)
        indexes_gpu = []
        for s in self.indexes:
            if isinstance(s, numpy.ndarray):
                s = cupy.array(s)
            indexes_gpu.append(s)
        indexes_gpu = tuple(indexes_gpu)
        b = a[indexes_gpu]
        b_cpu = a.get()[self.indexes]
        testing.assert_array_equal(b, b_cpu)

    def test_setitem(self):
        a_cpu = numpy.arange(numpy.prod(self.shape)).reshape(self.shape)
        a = cupy.array(a_cpu)
        indexes_gpu = []
        for s in self.indexes:
            if isinstance(s, numpy.ndarray):
                s = cupy.array(s)
            indexes_gpu.append(s)
        indexes_gpu = tuple(indexes_gpu)
        a[indexes_gpu] = -1
        a_cpu[self.indexes] = -1
        testing.assert_array_equal(a, a_cpu)


@testing.parameterize(
    {'shape': (), 'indexes': (-1,)},
    {'shape': (), 'indexes': (0,)},
    {'shape': (), 'indexes': (1,)},
    {'shape': (), 'indexes': ([0],)},
    {'shape': (), 'indexes': (numpy.array([0]),)},
    {'shape': (), 'indexes': (numpy.array(0),)},
    {'shape': (), 'indexes': numpy.array([True])},
    {'shape': (), 'indexes': numpy.array([False, True, True])},
    {'shape': (), 'indexes': ([False],)},
    {'shape': (0,), 'indexes': (-1,)},
    {'shape': (0,), 'indexes': (0,)},
    {'shape': (0,), 'indexes': (1,)},
    {'shape': (0,), 'indexes': ([0],)},
    {'shape': (0,), 'indexes': (numpy.array([0]),)},
    {'shape': (0,), 'indexes': (numpy.array(0),)},
    {'shape': (0,), 'indexes': numpy.array([True])},
    {'shape': (0,), 'indexes': numpy.array([False, True, True])},
    {'shape': (0, 1), 'indexes': (0, Ellipsis)},
    {'shape': (2, 3), 'indexes': (slice(None), [1, 2], slice(None))},
    {'shape': (2, 3), 'indexes': numpy.array([], dtype=numpy.float64)},
    {'shape': (3, 4), 'indexes': ([1, 0], [True, True])},
    {'shape': (2, 3, 4),
     'indexes': ([True, True], [[True, True, False, False]])},
    {'shape': (2, 3, 4),
     'indexes': ([True, True], [[True], [True], [False]])},
    {'shape': (2, 3, 4), 'indexes': numpy.empty((0, 1), bool)},
    {'shape': (2, 3, 4),
     'indexes': (numpy.empty(0, bool), numpy.empty((0, 2), bool))},
)
@testing.gpu
class TestArrayInvalidIndexAdvGetitem:

    def test_invalid_adv_getitem(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp)
            with pytest.raises(IndexError):
                a[self.indexes]


@testing.parameterize(
    {'shape': (0,), 'indexes': ([False],)},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), numpy.random.choice([False, True], (3, 1)))},
    {'shape': (2, 3, 4),
     'indexes': numpy.random.choice([False, True], (1, 3))},
    _ids=False,  # Do not generate ids from randomly generated params
)
@testing.gpu
class TestArrayInvalidIndexAdvGetitem2:

    def test_invalid_adv_getitem(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp)
            with pytest.raises(IndexError):
                a[self.indexes]


@testing.parameterize(
    {'shape': (2, 3, 4), 'indexes': [1, [1, [1]]]},
)
@testing.gpu
@testing.with_requires('numpy>=1.16')
class TestArrayInvalidValueAdvGetitem:

    def test_invalid_adv_getitem(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp)
            with pytest.raises(IndexError):
                with testing.assert_warns(FutureWarning):
                    a[self.indexes]


@testing.parameterize(
    # array only
    {'shape': (2, 3, 4), 'indexes': numpy.array(-1), 'value': 1},
    {'shape': (2, 3, 4), 'indexes': numpy.array([1, 0]), 'value': 1},
    {'shape': (2, 3, 4), 'indexes': [1, 0], 'value': 1},
    {'shape': (2, 3, 4), 'indexes': [1, -1], 'value': 1},
    {'shape': (2, 3, 4), 'indexes': (slice(None), [1, 2]), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), [[1, 2], [0, -1]],), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), slice(None), [[1, 2], [0, -1]]), 'value': 1},
    # slice and array
    {'shape': (2, 3, 4),
     'indexes': (slice(None), slice(1, 2), [[1, 2], [0, -1]]), 'value': 1},
    # None and array
    {'shape': (2, 3, 4),
     'indexes': (None, [1, -1]), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (None, [1, -1], None), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (None, None, None, [1, -1]), 'value': 1},
    # None, slice and array
    {'shape': (2, 3, 4),
     'indexes': (slice(0, 1), None, [1, -1]), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (slice(0, 1), slice(1, 2), [1, -1]), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (slice(0, 1), None, slice(1, 2), [1, -1]), 'value': 1},
    # mask
    {'shape': (2, 3, 4),
     'indexes': numpy.array([True, False]), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (1, numpy.array([True, False, True])), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (numpy.array([True, False]), 1), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), numpy.array([True, False, True])), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), 2, numpy.array([True, False, True, False])),
     'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), 2, False), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), slice(None),
                 numpy.random.choice([False, True], (4,))),
     'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (numpy.random.choice([False, True], (2, 3)),), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), numpy.random.choice([False, True], (3, 4)),),
     'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (numpy.random.choice([False, True], (2, 3, 4)),), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (1, None, numpy.array([True, False, True])), 'value': 1},
    # multiple arrays
    {'shape': (2, 3, 4), 'indexes': ([0, -1], [1, -1]), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': ([0, -1], [1, -1], [2, 1]), 'value': 1},
    {'shape': (2, 3, 4), 'indexes': ([0, -1], 1), 'value': 1},
    {'shape': (2, 3, 4), 'indexes': ([0, -1], slice(None), [1, -1]),
     'value': 1},
    {'shape': (2, 3, 4), 'indexes': ([0, -1], 1, 2), 'value': 1},
    {'shape': (2, 3, 4), 'indexes': ([1, 0], slice(None), [[2, 0], [3, 1]]),
     'value': 1},
    # multiple arrays and basic indexing
    {'shape': (2, 3, 4), 'indexes': ([0, -1], None, [1, 0]), 'value': 1},
    {'shape': (2, 3, 4), 'indexes': ([0, -1], slice(0, 2), [1, 0]),
     'value': 1},
    {'shape': (2, 3, 4), 'indexes': ([0, -1], None, slice(0, 2), [1, 0]),
     'value': 1},
    {'shape': (1, 1, 2, 3, 4),
     'indexes': (None, slice(None), slice(None), [1, 0], [2, -1], 1),
     'value': 1},
    {'shape': (1, 1, 2, 3, 4),
     'indexes': (None, slice(None), 0, [1, 0], slice(0, 2, 2), [2, -1]),
     'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), [0, -1], [[1, 0], [0, 1], [-1, 1]]), 'value': 1},
    # empty arrays
    {'shape': (2, 3, 4), 'indexes': [], 'value': 1},
    {'shape': (2, 3, 4), 'indexes': [],
     'value': numpy.array([1, 1, 1, 1])},
    {'shape': (2, 3, 4), 'indexes': [],
     'value': numpy.random.uniform(size=(3, 4))},
    {'shape': (2, 3, 4), 'indexes': numpy.array([], dtype=numpy.int32),
     'value': 1},
    {'shape': (2, 3, 4), 'indexes': numpy.array([[]], dtype=numpy.int32),
     'value': numpy.random.uniform(size=(3, 4))},
    {'shape': (2, 3, 4), 'indexes': (slice(None), []),
     'value': 1},
    {'shape': (2, 3, 4), 'indexes': ([], []),
     'value': 1},
    {'shape': (2, 3, 4), 'indexes': numpy.array([], dtype=numpy.bool_),
     'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), numpy.array([], dtype=numpy.bool_)),
     'value': 1},
    {'shape': (2, 3, 4), 'indexes': numpy.array([[], []], dtype=numpy.bool_),
     'value': numpy.random.uniform(size=(4,))},
    {'shape': (2, 3, 4), 'indexes': numpy.empty((0, 0, 4), bool), 'value': 1},
    # multiple masks
    {'shape': (2, 3, 4), 'indexes': (True, [True, False]), 'value': 1},
    {'shape': (2, 3, 4), 'indexes': (False, [True, False]), 'value': 1},
    {'shape': (2, 3, 4), 'indexes': (True, [[1]], slice(1, 2)), 'value': 1},
    {'shape': (2, 3, 4), 'indexes': (False, [[1]], slice(1, 2)), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (True, [[1]], slice(1, 2), True), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (True, [[1]], slice(1, 2), False), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (Ellipsis, [[1, 1, -3], [0, 2, 2]], [True, False, True, True]),
     'value': [[1, 2, 3], [4, 5, 6]]},
    {'shape': (2, 3, 4),
     'indexes': (numpy.empty((0, 3), bool), numpy.empty(0, bool)),
     'value': 1},
    # zero-dim and zero-sized arrays
    {'shape': (), 'indexes': Ellipsis, 'value': 1},
    {'shape': (), 'indexes': (), 'value': 1},
    {'shape': (), 'indexes': None, 'value': 1},
    {'shape': (), 'indexes': True, 'value': 1},
    {'shape': (), 'indexes': (True,), 'value': 1},
    {'shape': (), 'indexes': (False, True, True), 'value': 1},
    {'shape': (), 'indexes': numpy.ones((), dtype=numpy.bool_), 'value': 1},
    {'shape': (), 'indexes': numpy.zeros((), dtype=numpy.bool_), 'value': 1},
    {'shape': (0,), 'indexes': None, 'value': 1},
    {'shape': (0,), 'indexes': (), 'value': 1},
    {'shape': (0,), 'indexes': True, 'value': 1},
    {'shape': (0,), 'indexes': (True,), 'value': 1},
    {'shape': (0,), 'indexes': (False, True, True), 'value': 1},
    {'shape': (0,), 'indexes': numpy.ones((), dtype=numpy.bool_), 'value': 1},
    {'shape': (0,), 'indexes': numpy.zeros((), dtype=numpy.bool_), 'value': 1},
    # ellipsis
    {'shape': (2, 3, 4), 'indexes': (1, Ellipsis, 2), 'value': 1},
    # issue #1512
    {'shape': (2, 3, 4),
     'indexes': (Ellipsis, numpy.array(False)), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (Ellipsis, numpy.ones((3, 4), bool)), 'value': 1},
    # issue #4799
    {'shape': (3, 4, 5),
     'indexes': (slice(None), [0, 1], Ellipsis, [0, 1]), 'value': 1},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), [1, 0], Ellipsis, numpy.ones((5, 2), int)),
     'value': 1},
    _ids=False,  # Do not generate ids from randomly generated params
)
class TestArrayAdvancedIndexingSetitemScalarValue:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_setitem(self, xp, dtype):
        a = xp.zeros(self.shape, dtype=dtype)
        a[self.indexes] = self.value
        return a


@testing.parameterize(
    # empty arrays (list indexes)
    {'shape': (2, 3, 4), 'indexes': [[]], 'value': 1},
    {'shape': (2, 3, 4), 'indexes': [[[]]], 'value': 1},
    {'shape': (2, 3, 4), 'indexes': [[[[]]]], 'value': 1},
    {'shape': (2, 3, 4, 5), 'indexes': [[[[]]]], 'value': 1},
    {'shape': (2, 3, 4, 5), 'indexes': [[[[[]]]]], 'value': 1},
    # list indexes
    {'shape': (2, 3, 4), 'indexes': [[1]], 'value': 1},
    {'shape': (2, 3, 4), 'indexes': [[1, 0]], 'value': 1},
    {'shape': (2, 3, 4), 'indexes': [[1], [0]], 'value': 1},
)
@testing.with_requires('numpy>=1.23')
class TestArrayAdvancedIndexingSetitemScalarValue2:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_setitem(self, xp, dtype):
        a = xp.zeros(self.shape, dtype=dtype)
        a[self.indexes] = self.value
        return a


@testing.parameterize(
    # zero-dim and zero-sized arrays
    {'shape': (), 'indexes': numpy.array([True]), 'value': 1},
    {'shape': (), 'indexes': numpy.array([False, True, True]), 'value': 1},
    {'shape': (0,), 'indexes': numpy.array([True]), 'value': 1},
    {'shape': (0,), 'indexes': numpy.array([False, True, True]), 'value': 1},
)
@testing.gpu
class TestArrayAdvancedIndexingSetitemScalarValueIndexError:

    def test_adv_setitem(self):
        for xp in (numpy, cupy):
            a = xp.zeros(self.shape)
            with pytest.raises(IndexError):
                a[self.indexes] = self.value


@testing.parameterize(
    # list indexes
    {'shape': (2, 3, 4), 'indexes': [[1, 0], 2], 'value': 1},
    {'shape': (2, 3, 4), 'indexes': [[1], slice(1, 2)], 'value': 1},
    {'shape': (2, 3, 4), 'indexes': [[[1]], slice(1, 2)], 'value': 1},
)
@testing.with_requires('numpy>=1.23')
class TestArrayAdvancedIndexingSetitemScalarValueIndexError2:

    @testing.for_all_dtypes()
    def test_adv_setitem(self, dtype):
        for xp in (numpy, cupy):
            a = xp.zeros(self.shape, dtype=dtype)
            with pytest.raises(IndexError):
                with testing.assert_warns(numpy.VisibleDeprecationWarning):
                    a[self.indexes] = self.value


@testing.parameterize(
    {'shape': (2, 3, 4), 'indexes': numpy.array(1),
     'value': numpy.array([1])},
    {'shape': (2, 3, 4), 'indexes': numpy.array(1),
     'value': numpy.array([1, 2, 3, 4])},
    {'shape': (2, 3, 4), 'indexes': (slice(None), [0, -1]),
     'value': numpy.arange(2 * 2 * 4).reshape(2, 2, 4)},
    {'shape': (2, 5, 4), 'indexes': (slice(None), [[0, 2], [1, -1]]),
     'value': numpy.arange(2 * 2 * 2 * 4).reshape(2, 2, 2, 4)},
    # mask
    {'shape': (2, 3, 4), 'indexes': numpy.random.choice([False, True], (2, 3)),
     'value': numpy.arange(4)},
    {'shape': (2, 3, 4),
     'indexes': (slice(None), numpy.array([True, False, True])),
     'value': numpy.arange(2 * 2 * 4).reshape(2, 2, 4)},
    {'shape': (2, 3, 4),
     'indexes': (numpy.array([[True, False, False], [False, True, True]]),),
     'value': numpy.arange(3 * 4).reshape(3, 4)},
    {'shape': (2, 2, 2),
     'indexes': (slice(None), numpy.array([[True, False], [False, True]]),),
     'value': numpy.arange(2 * 2).reshape(2, 2)},
    {'shape': (2, 2, 2),
     'indexes': (numpy.array(
         [[[True, False], [True, False]], [[True, True], [False, False]]]),),
     'value': numpy.arange(4)},
    {'shape': (5,),
     'indexes': numpy.array([True, False, False, True, True]),
     'value': numpy.arange(3)},
    # multiple arrays
    {'shape': (2, 3, 4), 'indexes': ([1, 0], [2, 1]),
     'value': numpy.arange(2 * 4).reshape(2, 4)},
    {'shape': (2, 3, 4), 'indexes': ([1, 0], slice(None), [2, 1]),
     'value': numpy.arange(2 * 3).reshape(2, 3)},
    {'shape': (2, 3, 4), 'indexes': ([1, 0], slice(None), [[2, 0], [3, 1]]),
     'value': numpy.arange(2 * 2 * 3).reshape(2, 2, 3)},
    {'shape': (2, 3, 4),
     'indexes': ([[1, 0], [1, 0]], slice(None), [[2, 0], [3, 1]]),
     'value': numpy.arange(2 * 2 * 3).reshape(2, 2, 3)},
    {'shape': (2, 3, 4),
     'indexes': (1, slice(None), [[2, 0], [3, 1]]),
     'value': numpy.arange(2 * 2 * 3).reshape(2, 2, 3)},
    # list indexes
    {'shape': (2, 3, 4), 'indexes': [1],
     'value': numpy.arange(3 * 4).reshape(3, 4)},
    _ids=False,  # Do not generate ids from randomly generated params
)
@testing.gpu
class TestArrayAdvancedIndexingVectorValue:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_adv_setitem(self, xp, dtype):
        a = xp.zeros(self.shape, dtype=dtype)
        a[self.indexes] = self.value.astype(a.dtype)
        return a


@testing.gpu
class TestArrayAdvancedIndexingSetitemCupyIndices:

    shape = (2, 3)

    def test_cupy_indices_integer_array_1(self):
        a = cupy.zeros(self.shape)
        index = cupy.array([0, 1])
        original_index = index.copy()
        a[:, index] = cupy.array(1.)
        testing.assert_array_equal(
            a, cupy.array([[1., 1., 0.], [1., 1., 0.]]))
        testing.assert_array_equal(index, original_index)

    def test_cupy_indices_integer_array_2(self):
        a = cupy.zeros(self.shape)
        index = cupy.array([3, -5])
        original_index = index.copy()
        a[:, index] = cupy.array(1.)
        testing.assert_array_equal(
            a, cupy.array([[1., 1., 0.], [1., 1., 0.]]))
        testing.assert_array_equal(index, original_index)

    def test_cupy_indices_integer_array_3(self):
        a = cupy.zeros(self.shape)
        index = cupy.array([3, -5])
        original_index = index.copy()
        a[[1, 1], index] = cupy.array(1.)
        testing.assert_array_equal(
            a, cupy.array([[0., 0., 0.], [1., 1., 0.]]))
        testing.assert_array_equal(index, original_index)

    def test_cupy_indices_boolean_array(self):
        a = cupy.zeros(self.shape)
        index = cupy.array([True, False])
        original_index = index.copy()
        a[index] = cupy.array(1.)
        testing.assert_array_equal(
            a, cupy.array([[1., 1., 1.], [0., 0., 0.]]))
        testing.assert_array_almost_equal(original_index, index)


@testing.gpu
class TestArrayAdvancedIndexingSetitemDifferentDtypes:

    @testing.for_all_dtypes_combination(names=['src_dtype', 'dst_dtype'],
                                        no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_differnt_dtypes(self, xp, src_dtype, dst_dtype):
        shape = (2, 3)
        a = xp.zeros(shape, dtype=src_dtype)
        indexes = xp.array([0, 1])
        a[:, indexes] = xp.array(1, dtype=dst_dtype)
        return a

    @testing.for_all_dtypes_combination(names=['src_dtype', 'dst_dtype'],
                                        no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_differnt_dtypes_mask(self, xp, src_dtype, dst_dtype):
        shape = (2, 3)
        a = xp.zeros(shape, dtype=src_dtype)
        indexes = xp.array([True, False])
        a[indexes] = xp.array(1, dtype=dst_dtype)
        return a


@testing.gpu
class TestArrayAdvancedIndexingSetitemTranspose:

    @testing.numpy_cupy_array_equal()
    def test_adv_setitem_transp(self, xp):
        shape = (2, 3, 4)
        a = xp.zeros(shape).transpose(0, 2, 1)
        slices = (numpy.array([1, 0]), slice(None), numpy.array([2, 1]))
        a[slices] = 1
        return a
