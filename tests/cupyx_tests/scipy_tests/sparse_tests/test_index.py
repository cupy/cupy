import unittest

import cupy
from cupy import sparse
from cupy import testing

import numpy

import pytest


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.1, 0.4, 0.9],
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n_rows': [25, 150],
    'n_cols': [25, 150]
}))
@testing.with_requires('scipy')
@testing.gpu
class TestIndexing(unittest.TestCase):

    def _run(self, maj, min=None):
        a = sparse.random(self.n_rows, self.n_cols,
                          format=self.format,
                          density=self.density)

        # sparse.random doesn't support complex types
        # so we need to cast
        a = a.astype(self.dtype)

        maj_h = maj.get() if isinstance(maj, cupy.ndarray) else maj
        min_h = min.get() if isinstance(min, cupy.ndarray) else min

        expected = a.get()

        if min is not None:
            expected = expected[maj_h, min_h]
            actual = a[maj, min]
        else:
            expected = expected[maj_h]
            actual = a[maj]

        if sparse.isspmatrix(actual):
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
                actual.ravel(), numpy.array(expected).ravel())

    def test_major_slice(self):
        self._run(slice(5, 9))
        self._run(slice(9, 5))

    def test_major_all(self):
        self._run(slice(None))

    def test_major_scalar(self):
        self._run(10)

    def test_major_slice_minor_slice(self):
        self._run(slice(1, 5), slice(1, 5))

    def test_major_slice_minor_all(self):
        self._run(slice(1, 5), slice(None))
        self._run(slice(5, 1), slice(None))

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
