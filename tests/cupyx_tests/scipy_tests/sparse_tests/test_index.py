import unittest

import cupy

from cupy import sparse
from cupy import testing

import pytest


@testing.parameterize(*testing.product({
    'format': ['csr'],  # Taking out CSC until minor_fancy is included
    'density': [0.1, 0.4, 0.9],
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n_rows': [25, 150],
    'n_cols': [25, 150]
}))
@testing.with_requires('scipy>=1.4.0')
@testing.gpu
class TestIndexing(unittest.TestCase):

    def _run(self, maj, min=None, format=None):

        # Skipping tests that are only supported in one
        # format for now.
        if format is not None and format != self.format:
            pytest.skip()

        a = sparse.random(self.n_rows, self.n_cols,
                          format=self.format,
                          density=self.density)

        # sparse.random doesn't support complex types
        # so we need to cast
        a = a.astype(self.dtype)

        expected = a.get()

        maj_h = maj.get() if isinstance(maj, cupy.ndarray) else maj
        min_h = min.get() if isinstance(min, cupy.ndarray) else min

        if min is not None:
            expected = expected[maj_h, min_h]
            actual = a[maj, min]
        else:
            expected = expected[maj_h]
            actual = a[maj]

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
                actual, expected)

    # 2D Slicing

    def test_major_slice(self):
        self._run(slice(5, 9))
        self._run(slice(9, 5))

    def test_major_all(self):
        self._run(slice(None))

    def test_major_scalar(self):
        self._run(10)
        self._run(-10)

    def test_major_slice_minor_slice(self):
        self._run(slice(1, 5), slice(1, 5))

    def test_major_slice_minor_all(self):
        self._run(slice(1, 5), slice(None))
        self._run(slice(5, 1), slice(None))

    def test_major_slice_with_step(self):

        # CSR Tests
        self._run(slice(1, 20, 2), slice(1, 5, 1),
                  format='csr')
        self._run(slice(20, 1, 2), slice(1, 5, 1),
                  format='csr')
        self._run(slice(1, 15, 2), slice(1, 5, 1),
                  format='csr')
        self._run(slice(15, 1, 5), slice(1, 5, 1),
                  format='csr')
        self._run(slice(1, 15, 5), slice(1, 5, 1),
                  format='csr')
        self._run(slice(20, 1, 5), slice(None),
                  format='csr')
        self._run(slice(1, 20, 5), slice(None),
                  format='csr')

        # CSC Tests
        self._run(slice(1, 5, 1), slice(1, 20, 2),
                  format='csc')
        self._run(slice(1, 5, 1), slice(20, 1, 2),
                  format='csc')
        self._run(slice(1, 5, 1), slice(1, 15, 2),
                  format='csc')
        self._run(slice(1, 5, 1), slice(15, 1, 5),
                  format='csc')
        self._run(slice(None), slice(20, 1, 5),
                  format='csc')
        self._run(slice(None), slice(1, 20, 5),
                  format='csc')

    def test_major_scalar_minor_slice(self):
        self._run(5, slice(1, 5))

    def test_major_scalar_minor_all(self):
        self._run(5, slice(None))

    def test_major_scalar_minor_scalar(self):
        self._run(5, 5)

    def test_major_all_minor_scalar(self):
        self._run(slice(None), 5)

    def test_major_all_minor_slice(self):
        self._run(slice(None), slice(5, 10))

    def test_major_all_minor_all(self):
        self._run(slice(None), slice(None))

    # Major Indexing

    def test_major_bool_fancy(self):
        rand_bool = cupy.random.random(self.n_rows).astype(cupy.bool)
        self._run(rand_bool)

    def test_major_fancy_minor_all(self):
        self._run([1, 5, 4, 2, 5, 1], slice(None))

    def test_major_fancy_minor_scalar(self):
        self._run([1, 5, 4, 5, 1], 5)

    def test_major_fancy_minor_slice(self):
        self._run([1, 5, 4, 5, 1], slice(1, 5))
        self._run([1, 5, 4, 5, 1], slice(5, 1, 1))

    def test_ellipsis(self):
        self._run(Ellipsis)
        self._run(Ellipsis, 1)
        self._run(1, Ellipsis)
        self._run(Ellipsis, slice(None))
        self._run(slice(None), Ellipsis)
        self._run(Ellipsis, slice(1, None))
        self._run(slice(1, None), Ellipsis)

    def test_bad_indexing(self):
        with pytest.raises(IndexError):
            self._run("foo")

        with pytest.raises(IndexError):
            self._run(2, "foo")

        with pytest.raises(ValueError):
            self._run([1, 2, 3], [1, 2, 3, 4])
