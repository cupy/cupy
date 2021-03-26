import itertools
import unittest

import numpy
import pytest
try:
    import scipy.sparse
except ImportError:
    pass

import cupy
import cupyx
from cupy import testing
from cupyx.scipy import sparse


def _get_index_combos(idx):
    return [dict['arr_fn'](idx, dtype=dict['dtype'])
            for dict in testing.product({
                "arr_fn": [numpy.array, cupy.array],
                "dtype": [numpy.int32, numpy.int64]
            })]


def _check_shares_memory(xp, sp, x, y):
    if sp.issparse(x) and sp.issparse(y):
        assert not xp.shares_memory(x.indptr, y.indptr)
        assert not xp.shares_memory(x.indices, y.indices)
        assert not xp.shares_memory(x.data, y.data)


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.9],
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n_rows': [25, 150],
    'n_cols': [25, 150]
}))
@testing.with_requires('scipy>=1.4.0')
@testing.gpu
class TestSetitemIndexing(unittest.TestCase):

    def _run(self, maj, min=None, data=5):

        import scipy.sparse
        for i in range(2):
            shape = self.n_rows, self.n_cols
            a = testing.shaped_sparse_random(
                shape, sparse, self.dtype, self.density, self.format)
            expected = testing.shaped_sparse_random(
                shape, scipy.sparse, self.dtype, self.density, self.format)

            maj_h = maj.get() if isinstance(maj, cupy.ndarray) else maj
            min_h = min.get() if isinstance(min, cupy.ndarray) else min

            data_is_cupy = isinstance(data, (cupy.ndarray, sparse.spmatrix))
            data_h = data.get() if data_is_cupy else data

            if min is not None:
                actual = a
                actual[maj, min] = data

                expected[maj_h, min_h] = data_h
            else:
                actual = a
                actual[maj] = data

                expected[maj_h] = data_h

        if cupyx.scipy.sparse.isspmatrix(actual):
            actual.sort_indices()
            expected.sort_indices()

            cupy.testing.assert_array_equal(
                actual.indptr, expected.indptr)
            cupy.testing.assert_array_equal(
                actual.indices, expected.indices)
            cupy.testing.assert_array_equal(
                actual.data, expected.data)

        else:

            cupy.testing.assert_array_equal(
                actual.ravel(), cupy.array(expected).ravel())

    def test_set_sparse(self):

        x = cupyx.scipy.sparse.random(1, 5, format='csr', density=0.8)

        # Test inner indexing with sparse data
        for maj, min in zip(_get_index_combos([0, 1, 2, 3, 5]),
                            _get_index_combos([1, 2, 3, 4, 5])):
            self._run(maj, min, data=x)
        self._run([0, 1, 2, 3, 5], [1, 2, 3, 4, 5], data=x)

        # Test 2d major indexing 1d minor indexing with sparse data
        for maj, min in zip(_get_index_combos([[0], [1], [2], [3], [5]]),
                            _get_index_combos([1, 2, 3, 4, 5])):
            self._run(maj, min, data=x)
        self._run([[0], [1], [2], [3], [5]], [1, 2, 3, 4, 5], data=x)

        # Test 1d major indexing 2d minor indexing with sparse data
        for maj, min in zip(_get_index_combos([0, 1, 2, 3, 5]),
                            _get_index_combos([[1], [2], [3], [4], [5]])):
            self._run(maj, min, data=x)
        self._run([0, 1, 2, 3, 5], [[1], [2], [3], [4], [5]], data=x)

        # Test minor indexing numpy scalar / cupy 0-dim
        for maj, min in zip(_get_index_combos([0, 2, 4, 5, 6]),
                            _get_index_combos(1)):
            self._run(maj, min, data=x)

    @testing.with_requires('scipy>=1.5.0')
    def test_set_zero_dim_bool_mask(self):

        zero_dim_data = [numpy.array(5), cupy.array(5)]

        for data in zero_dim_data:
            self._run([False, True], data=data)

    def test_set_zero_dim_scalar(self):

        zero_dim_data = [numpy.array(5), cupy.array(5)]

        for data in zero_dim_data:
            self._run(slice(5, 10000), data=data)
            self._run([1, 5, 4, 5], data=data)
            self._run(0, 2, data=data)

    def test_major_slice(self):
        self._run(slice(5, 10000), data=5)
        self._run(slice(5, 4), data=5)
        self._run(slice(4, 5, 2), data=5)
        self._run(slice(5, 4, -2), data=5)
        self._run(slice(2, 4), slice(0, 2), [[4], [1]])
        self._run(slice(2, 4), slice(0, 2), [[4, 5], [6, 7]])
        self._run(slice(2, 4), 0, [[4], [6]])

        self._run(slice(5, 9))
        self._run(slice(9, 5))

    def test_major_all(self):
        self._run(slice(None))

    def test_major_scalar(self):
        self._run(10)

    def test_major_fancy(self):
        self._run([1, 5, 4])
        self._run([10, 2])
        self._run([2])

    def test_major_slice_minor_slice(self):
        self._run(slice(1, 5), slice(1, 5))

    def test_major_slice_minor_all(self):
        self._run(slice(1, 5), slice(None))
        self._run(slice(5, 1), slice(None))

    def test_major_slice_minor_scalar(self):
        self._run(slice(1, 5), 5)
        self._run(slice(5, 1), 5)
        self._run(slice(5, 1, -1), 5)

    def test_major_slice_minor_fancy(self):
        self._run(slice(1, 10, 2), [1, 5, 4])

    def test_major_scalar_minor_slice(self):
        self._run(5, slice(1, 5))

    def test_major_scalar_minor_all(self):
        self._run(5, slice(None))

    def test_major_scalar_minor_scalar(self):
        self._run(5, 5)
        self._run(10, 24, 5)

    def test_major_scalar_minor_fancy(self):
        self._run(5, [1, 5, 4])

    def test_major_all_minor_all(self):
        self._run(slice(None), slice(None))

    def test_major_all_minor_fancy(self):
        for min in _get_index_combos(
                [0, 3, 4, 1, 1, 5, 5, 2, 3, 4, 5, 4, 1, 5]):
            self._run(slice(None), min)

        self._run(slice(None), [0, 3, 4, 1, 1, 5, 5, 2, 3, 4, 5, 4, 1, 5])

    def test_major_fancy_minor_fancy(self):

        for maj, min in zip(_get_index_combos([1, 2, 3, 4, 1, 6, 1, 8, 9]),
                            _get_index_combos([1, 5, 2, 3, 4, 5, 4, 1, 5])):
            self._run(maj, min)

        self._run([1, 2, 3, 4, 1, 6, 1, 8, 9], [1, 5, 2, 3, 4, 5, 4, 1, 5])

        for idx in _get_index_combos([1, 5, 4]):
            self._run(idx, idx)
        self._run([1, 5, 4], [1, 5, 4])

        for maj, min in zip(_get_index_combos([2, 0, 10]),
                            _get_index_combos([9, 2, 1])):
            self._run(maj, min)
        self._run([2, 0, 10], [9, 2, 1])

        for maj, min in zip(_get_index_combos([2, 9]),
                            _get_index_combos([2, 1])):
            self._run(maj, min)
        self._run([2, 0], [2, 1])

    def test_major_fancy_minor_all(self):

        for idx in _get_index_combos([1, 5, 4]):
            self._run(idx, slice(None))
        self._run([1, 5, 4], slice(None))

    def test_major_fancy_minor_scalar(self):

        for idx in _get_index_combos([1, 5, 4]):
            self._run(idx, 5)
        self._run([1, 5, 4], 5)

    def test_major_fancy_minor_slice(self):

        for idx in _get_index_combos([1, 5, 4]):
            self._run(idx, slice(1, 5))
            self._run(idx, slice(5, 1, -1))
        self._run([1, 5, 4], slice(1, 5))
        self._run([1, 5, 4], slice(5, 1, -1))

    def test_major_bool_fancy(self):
        rand_bool = testing.shaped_random(self.n_rows, dtype=bool)
        self._run(rand_bool)

    def test_major_slice_with_step(self):

        # positive step
        self._run(slice(1, 10, 2))
        self._run(slice(2, 10, 5))
        self._run(slice(0, 10, 10))

        self._run(slice(1, None, 2))
        self._run(slice(2, None, 5))
        self._run(slice(0, None,  10))

        # negative step
        self._run(slice(10, 1, -2))
        self._run(slice(10, 2, -5))
        self._run(slice(10, 0, -10))

        self._run(slice(10, None, -2))
        self._run(slice(10, None, -5))
        self._run(slice(10, None, -10))

    def test_major_slice_with_step_minor_slice_with_step(self):

        # positive step
        self._run(slice(1, 10, 2), slice(1, 10, 2))
        self._run(slice(2, 10, 5), slice(2, 10, 5))
        self._run(slice(0, 10, 10), slice(0, 10, 10))

        # negative step
        self._run(slice(10, 1, 2), slice(10, 1, 2))
        self._run(slice(10, 2, 5), slice(10, 2, 5))
        self._run(slice(10, 0, 10), slice(10, 0, 10))

    def test_major_slice_with_step_minor_all(self):

        # positive step
        self._run(slice(1, 10, 2), slice(None))
        self._run(slice(2, 10, 5), slice(None))
        self._run(slice(0, 10, 10), slice(None))

        # negative step
        self._run(slice(10, 1, 2), slice(None))
        self._run(slice(10, 2, 5), slice(None))
        self._run(slice(10, 0, 10), slice(None))

    @testing.with_requires('scipy>=1.5.0')
    def test_fancy_setting_bool(self):
        # Unfortunately, boolean setting is implemented slightly
        # differently between Scipy 1.4 and 1.5. Using the most
        # up-to-date version in CuPy.

        for maj in _get_index_combos(
                [[True], [False], [False], [True], [True], [True]]):
            self._run(maj, data=5)
        self._run([[True], [False], [False], [True], [True], [True]], data=5)

        for maj in _get_index_combos([True, False, False, True, True, True]):
            self._run(maj, data=5)
        self._run([True, False, False, True, True, True], data=5)

        for maj in _get_index_combos([[True], [False], [True]]):
            self._run(maj, data=5)
        self._run([[True], [False], [True]], data=5)

    def test_fancy_setting(self):

        for maj, data in zip(_get_index_combos([0, 5, 10, 2]),
                             _get_index_combos([1, 2, 3, 2])):
            self._run(maj, 0, data)
        self._run([0, 5, 10, 2], 0, [1, 2, 3, 2])

        # Indexes with duplicates should follow 'last-in-wins'
        # But Cupy dense indexing doesn't support this yet:
        # ref: https://github.com/cupy/cupy/issues/3836
        # Starting with an empty array for now, since insertions
        # use `last-in-wins`.
        self.density = 0.0  # Zeroing out density to force only insertions
        for maj, min, data in zip(_get_index_combos([0, 5, 10, 2, 0, 10]),
                                  _get_index_combos([1, 2, 3, 4, 1, 3]),
                                  _get_index_combos([1, 2, 3, 4, 5, 6])):
            self._run(maj, min, data)


class IndexingTestBase(unittest.TestCase):

    def _make_matrix(self, sp, dtype):
        shape = self.n_rows, self.n_cols
        return testing.shaped_sparse_random(
            shape, sp, dtype, self.density, self.format)

    def _make_indices(self, xp, dtype=None):
        indices = []
        for ind in self.indices:
            if isinstance(ind, slice):
                indices.append(ind)
            else:
                indices.append(xp.array(ind, dtype=dtype))

        return tuple(indices)


_int_index = [0, -1, 10, -10]
_slice_index = [
    slice(0, 0), slice(None), slice(3, 17), slice(17, 3, -1)
]
_slice_index_full = [
    slice(0, 0), slice(0, 1), slice(5, 6), slice(None),
    slice(3, 17, 1), slice(17, 3, 1), slice(2, -1, 1), slice(-1, 2, 1),
    slice(3, 17, -1), slice(17, 3, -1), slice(2, -1, -1), slice(-1, 2, -1),
    slice(3, 17, 2), slice(17, 3, 2), slice(3, 17, -2), slice(17, 3, -2),
]
_int_array_index = [
    [], [0], [0, 0], [1, 5, 4, 5, 2, 4, 1]
]


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.0, 0.5],
    'n_rows': [1, 25],
    'n_cols': [1, 25],
    'indices': (
        # Int
        _int_index
        # Slice
        + _slice_index_full
        # Int x Int
        + list(itertools.product(_int_index, _int_index))
        # Slice x Slice
        + list(itertools.product(_slice_index, _slice_index))
        # Int x Slice
        + list(itertools.product(_int_index, _slice_index))
        + list(itertools.product(_slice_index, _int_index))
        # Ellipsis
        + [
            (Ellipsis,),
            (Ellipsis, slice(None)),
            (slice(None), Ellipsis),
            (Ellipsis, 1),
            (1, Ellipsis),
            (slice(1, None), Ellipsis),
            (Ellipsis, slice(1, None)),
        ]
    ),
}))
@testing.with_requires('scipy>=1.4.0')
@testing.gpu
class TestSliceIndexing(IndexingTestBase):

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_array_equal(
        sp_name='sp', type_check=False, accept_error=IndexError)
    def test_indexing(self, xp, sp, dtype):
        a = self._make_matrix(sp, dtype)
        res = a[self.indices]
        _check_shares_memory(xp, sp, a, res)
        return res


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.0, 0.5],
    'n_rows': [1, 25],
    'n_cols': [1, 25],
    'indices': (
        # Array
        _int_array_index
        # Array x Int
        + list(itertools.product(_int_array_index, _int_index))
        + list(itertools.product(_int_index, _int_array_index))
        # Array x Slice
        + list(itertools.product(_slice_index, _int_array_index))
        # SciPy chose inner indexing for int-array x slice inputs.
        # + list(itertools.product(_int_array_index, _slice_index))
        # Array x Array (Inner indexing)
        + [
            ([], []),
            ([0], [0]),
            ([1, 5, 4], [1, 5, 4]),
            ([2, 0, 10, 0, 2], [9, 2, 1, 0, 2]),
            ([2, 0, 10, 0], [9, 2, 1, 0]),
            ([2, 0, 2], [2, 1, 1]),
            ([2, 0, 2], [2, 1, 2]),
        ]
    )
}))
@testing.with_requires('scipy>=1.4.0')
@testing.gpu
class TestArrayIndexing(IndexingTestBase):

    def setUp(self):
        indices = self.indices
        if not isinstance(indices, tuple):
            indices = (indices,)
        for index, size in zip(indices, [self.n_rows, self.n_cols]):
            if isinstance(index, list):
                for ind in index:
                    if not (0 <= ind < size):
                        # CuPy does not check boundaries.
                        pytest.skip('Out of bounds')

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_array_equal(
        sp_name='sp', type_check=False, accept_error=IndexError)
    def test_list_indexing(self, xp, sp, dtype):
        a = self._make_matrix(sp, dtype)
        res = a[self.indices]
        _check_shares_memory(xp, sp, a, res)
        return res

    @testing.for_dtypes('fdFD')
    @testing.for_dtypes('il', name='ind_dtype')
    @testing.numpy_cupy_array_equal(
        sp_name='sp', type_check=False, accept_error=IndexError)
    def test_numpy_ndarray_indexing(self, xp, sp, dtype, ind_dtype):
        a = self._make_matrix(sp, dtype)
        indices = self._make_indices(numpy, ind_dtype)
        res = a[indices]
        _check_shares_memory(xp, sp, a, res)
        return res

    @testing.for_dtypes('fdFD')
    @testing.for_dtypes('il', name='ind_dtype')
    @testing.numpy_cupy_array_equal(
        sp_name='sp', type_check=False, accept_error=IndexError)
    def test_cupy_ndarray_indexing(self, xp, sp, dtype, ind_dtype):
        a = self._make_matrix(sp, dtype)
        indices = self._make_indices(xp, ind_dtype)
        res = a[indices]
        _check_shares_memory(xp, sp, a, res)
        return res


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.0, 0.5],
    'indices': [
        # Bool array x Int
        ([True, False, True], 3),
        (2, [True, False, True, False, True]),
        # Bool array x Slice
        ([True, False, True], slice(None)),
        ([True, False, True], slice(1, 4)),
        (slice(None), [True, False, True, False, True]),
        (slice(1, 4), [True, False, True, False, True]),
        # Bool array x Bool array
        # SciPy chose inner indexing for int-array x slice inputs.
        ([True, False, True], [True, False, True]),
    ],
}))
@testing.with_requires('scipy>=1.4.0')
@testing.gpu
class TestBoolMaskIndexing(IndexingTestBase):

    n_rows = 3
    n_cols = 5

    # In older environments (e.g., py35, scipy 1.4), scipy sparse arrays are
    # crashing when indexed with native Python boolean list.
    @testing.with_requires('scipy>=1.5.0')
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_array_equal(sp_name='sp', type_check=False)
    def test_bool_mask(self, xp, sp, dtype):
        a = self._make_matrix(sp, dtype)
        res = a[self.indices]
        _check_shares_memory(xp, sp, a, res)
        return res

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_array_equal(sp_name='sp', type_check=False)
    def test_numpy_bool_mask(self, xp, sp, dtype):
        a = self._make_matrix(sp, dtype)
        indices = self._make_indices(numpy)
        res = a[indices]
        _check_shares_memory(xp, sp, a, res)
        return res

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_array_equal(sp_name='sp', type_check=False)
    def test_cupy_bool_mask(self, xp, sp, dtype):
        a = self._make_matrix(sp, dtype)
        indices = self._make_indices(xp)
        res = a[indices]
        _check_shares_memory(xp, sp, a, res)
        return res


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.4],
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n_rows': [25, 150],
    'n_cols': [25, 150],
    'indices': [
        ('foo',),
        (2, 'foo'),
        ([[0, 0], [1, 1]]),
    ],
}))
@testing.with_requires('scipy>=1.4.0')
@testing.gpu
class TestIndexingIndexError(IndexingTestBase):

    def test_indexing_index_error(self):
        for xp, sp in [(numpy, scipy.sparse), (cupy, sparse)]:
            a = self._make_matrix(sp, numpy.float32)
            with pytest.raises(IndexError):
                a[self.indices]


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.4],
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n_rows': [25, 150],
    'n_cols': [25, 150],
    'indices': [
        ([1, 2, 3], [1, 2, 3, 4]),
    ],
}))
@testing.with_requires('scipy>=1.4.0')
@testing.gpu
class TestIndexingValueError(IndexingTestBase):

    def test_indexing_value_error(self):
        for xp, sp in [(numpy, scipy.sparse), (cupy, sparse)]:
            a = self._make_matrix(sp, numpy.float32)
            with pytest.raises(ValueError):
                a[self.indices]
