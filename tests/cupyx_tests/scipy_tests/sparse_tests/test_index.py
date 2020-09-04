import unittest

import numpy
import pytest
import scipy.sparse

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


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.1, 0.4, 0.9],
    'n_rows': [25, 150],
    'n_cols': [25, 150],
    'indices': [
        # Single
        (10,),
        (-10,),
        # Slice
        (slice(5, 9),),
        (slice(9, 5),),
        (slice(None),),
        # Int x Int
        (5, 5),
        # Slice x Slice
        (slice(1, 5), slice(1, 5)),
        (slice(1, 20, 2), slice(1, 5, 1)),
        (slice(20, 1, 2), slice(1, 5, 1)),
        (slice(1, 15, 2), slice(1, 5, 1)),
        (slice(15, 1, 5), slice(1, 5, 1)),
        (slice(1, 15, 5), slice(1, 5, 1)),
        (slice(1, 5), slice(None)),
        (slice(5, 1), slice(None)),
        (slice(20, 1, 5), slice(None)),
        (slice(1, 20, 5), slice(None)),
        (slice(1, 5, 1), slice(1, 20, 2)),
        (slice(1, 5, 1), slice(20, 1, 2)),
        (slice(1, 5, 1), slice(1, 15, 2)),
        (slice(1, 5, 1), slice(15, 1, 5)),
        (slice(1, 5, 1), slice(1, 15, 5)),
        (slice(None), slice(20, 1, 5)),
        (slice(None), slice(1, 20, 5)),
        (slice(1, 20, 2), slice(1, 20, 2)),
        (slice(None), slice(None)),
        (slice(1, 5), slice(None)),
        (slice(5, 1), slice(None)),
        # Slice x Int
        (slice(1, 5), 5),
        (slice(5, 1), 5),
        (slice(5, 1, -1), 5),
        (5, slice(1, 5)),
        (5, slice(5, 1)),
        (5, slice(5, 1, -1)),
        # Ellipsis
        (Ellipsis,),
        (Ellipsis, slice(None)),
        (slice(None), Ellipsis),
        (Ellipsis, 1),
        (1, Ellipsis),
        (slice(1, None), Ellipsis),
        (Ellipsis, slice(1, None)),
    ],
}))
@testing.with_requires('scipy>=1.4.0')
@testing.gpu
class TestSliceIndexing(IndexingTestBase):

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_array_equal(sp_name='sp', type_check=False)
    def test_indexing(self, xp, sp, dtype):
        a = self._make_matrix(sp, dtype)
        return a[self.indices]


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.1, 0.4, 0.9],
    'n_rows': [25, 150],
    'n_cols': [25, 150],
    'indices': [
        # 0-dim array
        (10,),
        (-10,),
        (5, slice(1, 5)),
        (5, slice(None)),
        (slice(None), 5),
        (5, 5),
        # Outer indexing
        ([1, 5, 4, 2, 5, 1], slice(None)),
        ([1, 5, 2, 3, 4, 5, 4, 1, 5], slice(None)),
        ([0, 3, 4, 1, 1, 5, 5, 2, 3, 4, 5, 4, 1, 5], slice(None)),
        ([1, 5, 4, 5, 2, 4, 1], slice(None)),
        (slice(None), [1, 5, 4, 2, 5, 1]),
        (slice(None), [1, 5, 2, 3, 4, 5, 4, 1, 5]),
        (slice(None), [0, 3, 4, 1, 1, 5, 5, 2, 3, 4, 5, 4, 1, 5]),
        (slice(None), [1, 5, 4, 5, 2, 4, 1]),
        ([1, 5, 4, 5, 1], 5),
        (5, [1, 5, 4, 5, 1]),
        ([1, 5, 4, 5, 1], slice(1, 5)),
        (slice(5, 1, 1), [1, 5, 4, 5, 1]),
        # TODO (asi1024): Support
        # ([1, 5, 4, 5, 2, 4, 1], slice(1, 10, 2)),
        (slice(1, 10, 2), [1, 5, 4, 5, 2, 4, 1]),
        # Inner indexing
        ([1, 5, 4], [1, 5, 4]),
        ([2, 0, 10, 0, 2], [9, 2, 1, 0, 2]),
        ([2, 0, 10, 0], [9, 2, 1, 0]),
        ([2, 0, 2], [2, 1, 1]),
        ([2, 0, 2], [2, 1, 2]),
    ],
}))
@testing.with_requires('scipy>=1.4.0')
@testing.gpu
class TestArrayIndexing(IndexingTestBase):

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_list_indexing(self, xp, sp, dtype):
        a = self._make_matrix(sp, dtype)
        return a[self.indices]

    @testing.for_dtypes('fdFD')
    @testing.for_dtypes('il', name='ind_dtype')
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_numpy_ndarray_indexing(self, xp, sp, dtype, ind_dtype):
        a = self._make_matrix(sp, dtype)
        indices = self._make_indices(numpy, ind_dtype)
        return a[indices]

    @testing.for_dtypes('fdFD')
    @testing.for_dtypes('il', name='ind_dtype')
    @testing.numpy_cupy_array_equal(sp_name='sp')
    def test_cupy_ndarray_indexing(self, xp, sp, dtype, ind_dtype):
        a = self._make_matrix(sp, dtype)
        indices = self._make_indices(xp, ind_dtype)
        print(indices)
        return a[indices]


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.1, 0.4, 0.9],
    'n_rows': [3],
    'n_cols': [5],
    'indices': [
        # Bool array x Int
        ([True, False, True], 3),
        (2, [True, False, True, False, True]),
        # Bool array x All
        ([True, False, True], slice(None)),
        (slice(None), [True, False, True, False, True]),
        # Bool array x Slice
        ([True, False, True], slice(1, 4)),
        (slice(1, 4), [True, False, True, False, True]),
        # Bool array x Bool array
        ([True, False, True], [True, False, True]),
    ],
}))
@testing.with_requires('scipy>=1.4.0')
@testing.gpu
class TestBoolMaskIndexing(IndexingTestBase):

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
