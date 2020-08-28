import unittest

import cupy

from cupy import testing
from cupyx.scipy import sparse

import numpy

import pytest


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.1, 0.4, 0.9],
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n_rows': [25, 150],
    'n_cols': [25, 150]
}))
@testing.with_requires('scipy>=1.4.0')
@testing.gpu
class TestIndexing(unittest.TestCase):

    def _run(self, maj, min=None, flip_for_csc=True,
             compare_dense=False):

        a = sparse.random(self.n_rows, self.n_cols,
                          format=self.format,
                          density=self.density)

        if self.format == 'csc' and flip_for_csc:
            tmp = maj
            maj = min
            min = tmp

        # None is not valid for major when minor is not None
        maj = slice(None) if maj is None else maj

        # sparse.random doesn't support complex types
        # so we need to cast
        a = a.astype(self.dtype)

        expected = a.get()

        if compare_dense:
            expected = expected.todense()

        maj_h = maj.get() if isinstance(maj, cupy.ndarray) else maj
        min_h = min.get() if isinstance(min, cupy.ndarray) else min

        if min is not None:

            actual = a[maj, min]
            expected = expected[maj_h, min_h]
        else:
            actual = a[maj]
            expected = expected[maj_h]

        if compare_dense:
            actual = actual.todense()

        if sparse.isspmatrix(actual):
            actual.sort_indices()
            expected.sort_indices()

            testing.assert_array_equal(
                actual.indptr, expected.indptr)
            testing.assert_array_equal(
                actual.indices, expected.indices)
            testing.assert_array_equal(
                actual.data, expected.data)
        else:
            testing.assert_array_equal(
                actual, numpy.asarray(expected))

    @staticmethod
    def _get_index_combos(idx):
        return [dict['arr_fn'](idx, dtype=dict['dtype'])
                for dict in testing.product({
                    "arr_fn": [numpy.array, cupy.array],
                    "dtype": [numpy.int32, numpy.int64]
                })]

    # 2D Slicing

    def test_major_slice(self):
        self._run(slice(5, 9))
        self._run(slice(9, 5))

    def test_major_all(self):
        self._run(slice(None))

    def test_major_scalar(self):
        self._run(10)
        self._run(-10)

        self._run(numpy.array(10))
        self._run(numpy.array(-10))

        self._run(cupy.array(10))
        self._run(cupy.array(-10))

    def test_major_slice_minor_slice(self):
        self._run(slice(1, 5), slice(1, 5))
        self._run(slice(1, 20, 2), slice(1, 5, 1))
        self._run(slice(20, 1, 2), slice(1, 5, 1))
        self._run(slice(1, 15, 2), slice(1, 5, 1))
        self._run(slice(15, 1, 5), slice(1, 5, 1))
        self._run(slice(1, 15, 5), slice(1, 5, 1))
        self._run(slice(20, 1, 5), slice(None))
        self._run(slice(1, 20, 5), slice(None))

    def test_major_slice_minor_all(self):
        self._run(slice(1, 5), slice(None))
        self._run(slice(5, 1), slice(None))

    def test_major_slice_minor_scalar(self):
        self._run(slice(1, 5), 5)
        self._run(slice(5, 1), 5)
        self._run(slice(5, 1, -1), 5)

    def test_major_scalar_minor_slice(self):
        self._run(5, slice(1, 5))
        self._run(numpy.array(5), slice(1, 5))
        self._run(cupy.array(5), slice(1, 5))

    def test_major_scalar_minor_all(self):
        self._run(5, slice(None))
        self._run(numpy.array(5), slice(None))

    def test_major_scalar_minor_scalar(self):
        self._run(5, 5)
        self._run(numpy.array(5), numpy.array(5))
        self._run(cupy.array(5), cupy.array(5))

    def test_major_all_minor_scalar(self):
        self._run(slice(None), 5)

    def test_major_all_minor_slice(self):
        self._run(slice(None), slice(5, 10))

    def test_major_all_minor_all(self):
        self._run(slice(None), slice(None))

    def test_ellipsis(self):
        self._run(Ellipsis, flip_for_csc=False)
        self._run(Ellipsis, 1, flip_for_csc=False)
        self._run(1, Ellipsis, flip_for_csc=False)
        self._run(Ellipsis, slice(None), flip_for_csc=False)
        self._run(slice(None), Ellipsis, flip_for_csc=False)
        self._run(Ellipsis, slice(1, None), flip_for_csc=False)
        self._run(slice(1, None), Ellipsis, flip_for_csc=False)

    # Major Indexing

    def test_major_bool_fancy(self):

        size = self.n_rows if self.format == 'csr' else self.n_cols

        a = numpy.random.random(size)
        self._run(cupy.array(a).astype(cupy.bool))  # Cupy
        self._run(a.astype(numpy.bool))             # Numpy
        self._run(a.astype(numpy.bool).tolist(),    # List
                  # In older environments (e.g., py35, scipy 1.4),
                  # scipy sparse arrays are crashing when indexed with
                  # native Python boolean list.
                  compare_dense=True)

    def test_major_fancy_minor_all(self):

        self._run([1, 5, 4, 2, 5, 1], slice(None))

        for idx in self._get_index_combos([1, 5, 4, 2, 5, 1]):
            self._run(idx, slice(None))

    def test_major_fancy_minor_scalar(self):
        self._run([1, 5, 4, 5, 1], 5)
        for idx in self._get_index_combos([1, 5, 4, 2, 5, 1]):
            self._run(idx, 5)

    def test_major_fancy_minor_slice(self):
        self._run([1, 5, 4, 5, 1], slice(1, 5))
        self._run([1, 5, 4, 5, 1], slice(5, 1, 1))

        for idx in self._get_index_combos([1, 5, 4, 5, 1]):
            self._run(idx, slice(5, 1, 1))

        for idx in self._get_index_combos([1, 5, 4, 5, 1]):
            self._run(idx, slice(1, 5))

    # Minor Indexing

    def test_major_all_minor_bool(self):
        size = self.n_cols if self.format == 'csr' else self.n_rows

        a = numpy.random.random(size)
        self._run(slice(None), cupy.array(a).astype(cupy.bool))  # Cupy
        self._run(slice(None), a.astype(numpy.bool))  # Numpy
        self._run(slice(None), a.astype(numpy.bool).tolist(),  # List
                  # In older environments (e.g., py35, scipy 1.4),
                  # scipy sparse arrays are crashing when indexed with
                  # native Python boolean list.
                  compare_dense=True)

    def test_major_slice_minor_bool(self):
        size = self.n_cols if self.format == 'csr' else self.n_rows

        a = numpy.random.random(size)
        self._run(slice(1, 10, 2), cupy.array(a).astype(cupy.bool))  # Cupy
        self._run(slice(1, 10, 2), a.astype(numpy.bool))  # Numpy
        self._run(slice(1, 10, 2), a.astype(numpy.bool).tolist(),  # List
                  # In older environments (e.g., py35, scipy 1.4),
                  # scipy sparse arrays are crashing when indexed with
                  # native Python boolean list.
                  compare_dense=True)

    def test_major_all_minor_fancy(self):

        self._run(slice(None), [1, 5, 2, 3, 4, 5, 4, 1, 5])
        self._run(slice(None), [0, 3, 4, 1, 1, 5, 5, 2, 3, 4, 5, 4, 1, 5])

        self._run(slice(None), [1, 5, 4, 5, 2, 4, 1])

        for idx in self._get_index_combos([1, 5, 4, 5, 2, 4, 1]):
            self._run(slice(None), idx)

    def test_major_slice_minor_fancy(self):

        self._run(slice(1, 10, 2), [1, 5, 4, 5, 2, 4, 1])

        for idx in self._get_index_combos([1, 5, 4, 5, 2, 4, 1]):
            self._run(slice(1, 10, 2), idx)

    def test_major_scalar_minor_fancy(self):

        self._run(5, [1, 5, 4, 1, 2])

        for idx in self._get_index_combos([1, 5, 4, 1, 2]):
            self._run(5, idx)

    # Inner Indexing

    def test_major_fancy_minor_fancy(self):

        for idx in self._get_index_combos([1, 5, 4]):
            self._run(idx, idx)

        self._run([1, 5, 4], [1, 5, 4])

        maj = self._get_index_combos([2, 0, 10, 0, 2])
        min = self._get_index_combos([9, 2, 1, 0, 2])

        for (idx1, idx2) in zip(maj, min):
            self._run(idx1, idx2)

        self._run([2, 0, 10, 0], [9, 2, 1, 0])

        maj = self._get_index_combos([2, 0, 2])
        min = self._get_index_combos([2, 1, 1])

        for (idx1, idx2) in zip(maj, min):
            self._run(idx1, idx2)

        self._run([2, 0, 2], [2, 1, 2])

    # Bad Indexing

    def test_bad_indexing(self):
        with pytest.raises(IndexError):
            self._run("foo")

        with pytest.raises(IndexError):
            self._run(2, "foo")

        with pytest.raises(ValueError):
            self._run([1, 2, 3], [1, 2, 3, 4])

        with pytest.raises(IndexError):
            self._run([[0, 0], [1, 1]])
