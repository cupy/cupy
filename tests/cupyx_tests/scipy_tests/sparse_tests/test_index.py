import itertools
import unittest

import numpy
import pytest
try:
    import scipy.sparse
except ImportError:
    pass

import cupy
from cupy import testing
from cupyx.scipy import sparse


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
        return a[self.indices]


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
        return a[self.indices]

    @testing.for_dtypes('fdFD')
    @testing.for_dtypes('il', name='ind_dtype')
    @testing.numpy_cupy_array_equal(
        sp_name='sp', type_check=False, accept_error=IndexError)
    def test_numpy_ndarray_indexing(self, xp, sp, dtype, ind_dtype):
        a = self._make_matrix(sp, dtype)
        indices = self._make_indices(numpy, ind_dtype)
        return a[indices]

    @testing.for_dtypes('fdFD')
    @testing.for_dtypes('il', name='ind_dtype')
    @testing.numpy_cupy_array_equal(
        sp_name='sp', type_check=False, accept_error=IndexError)
    def test_cupy_ndarray_indexing(self, xp, sp, dtype, ind_dtype):
        a = self._make_matrix(sp, dtype)
        indices = self._make_indices(xp, ind_dtype)
        print(indices)
        return a[indices]


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
        return a[self.indices]

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_array_equal(sp_name='sp', type_check=False)
    def test_numpy_bool_mask(self, xp, sp, dtype):
        a = self._make_matrix(sp, dtype)
        indices = self._make_indices(numpy)
        return a[indices]

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_array_equal(sp_name='sp', type_check=False)
    def test_cupy_bool_mask(self, xp, sp, dtype):
        a = self._make_matrix(sp, dtype)
        indices = self._make_indices(xp)
        return a[indices]


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
