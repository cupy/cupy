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

    def test_major_fancy(self):
        with pytest.raises(NotImplementedError):
            self._run([1, 5, 4])
            self._run([10, 2])
            self._run([2])

    def test_major_bool_fancy(self):
        with pytest.raises(NotImplementedError):
            rand_bool = cupy.random.random(self.n_rows).astype(cupy.bool)
            self._run(rand_bool)

    def test_major_slice_minor_slice(self):
        self._run(slice(1, 5), slice(1, 5))

    def test_major_slice_minor_all(self):
        self._run(slice(1, 5), slice(None))
        self._run(slice(5, 1), slice(None))

    # def test_major_slice_minor_scalar(self):
    #     self._run(slice(1, 5), 5)
    #     self._run(slice(5, 1), 5)
    #
    #     def do_run():
    #         self._run(slice(5, 1, -1), 5)
    #
    #     if self.format == 'csr':
    #         with pytest.raises(NotImplementedError):
    #             do_run()

    def test_major_slice_minor_fancy(self):
        with pytest.raises(NotImplementedError):
            self._run(slice(1, 10, 2), [1, 5, 4])

    def test_major_scalar_minor_slice(self):
        self._run(5, slice(1, 5))

    def test_major_scalar_minor_all(self):
        self._run(5, slice(None))

    def test_major_scalar_minor_scalar(self):
        self._run(5, 5)

    def test_major_scalar_minor_fancy(self):
        with pytest.raises(NotImplementedError):
            self._run(5, [1, 5, 4])

    def test_major_all_minor_scalar(self):
        self._run(slice(None), 5)

    def test_major_all_minor_slice(self):
        self._run(slice(None), slice(5, 10))

    def test_major_all_minor_all(self):
        self._run(slice(None), slice(None))

    def test_major_all_minor_fancy(self):
        with pytest.raises(NotImplementedError):
            self._run(slice(None), [1, 5, 2, 3, 4, 5, 4, 1, 5])
            self._run(slice(None), [0, 3, 4, 1, 1, 5, 5, 2, 3, 4, 5, 4, 1, 5])

    def test_major_fancy_minor_fancy(self):
        with pytest.raises(NotImplementedError):
            self._run([1, 5, 4], [1, 5, 4])
            self._run([2, 0, 10], [9, 2, 1])
            self._run([2, 0], [2, 1])

    def test_major_fancy_minor_all(self):
        with pytest.raises(NotImplementedError):
            self._run([1, 5, 4], slice(None))

    def test_major_fancy_minor_scalar(self):
        with pytest.raises(NotImplementedError):
            self._run([1, 5, 4], 5)

    def test_major_fancy_minor_slice(self):
        with pytest.raises(NotImplementedError):
            self._run([1, 5, 4], slice(1, 5))
            self._run([1, 5, 4], slice(5, 1, -1))

    def test_major_slice_with_step(self):

        def test_runs():

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

        if self.format == 'csc':
            with pytest.raises(NotImplementedError):
                test_runs()
        else:
            test_runs()

    # def test_major_slice_with_step_minor_slice_with_step(self):
    #
    #     with pytest.raises(NotImplementedError):
    #         # positive step
    #         self._run(slice(1, 10, 2), slice(1, 10, 2))
    #         self._run(slice(2, 10, 5), slice(2, 10, 5))
    #         self._run(slice(0, 10, 10), slice(0, 10, 10))
    #
    #         # negative step
    #         self._run(slice(10, 1, 2), slice(10, 1, 2))
    #         self._run(slice(10, 2, 5), slice(10, 2, 5))
    #         self._run(slice(10, 0, 10), slice(10, 0, 10))

    def test_major_slice_with_step_minor_all(self):
        def test_runs():
            # positive step
            self._run(slice(1, 10, 2), slice(None))
            self._run(slice(2, 10, 5), slice(None))
            self._run(slice(0, 10, 10), slice(None))

            # negative step
            self._run(slice(10, 1, 2), slice(None))
            self._run(slice(10, 2, 5), slice(None))
            self._run(slice(10, 0, 10), slice(None))

        if self.format == 'csc':
            with pytest.raises(NotImplementedError):
                test_runs()
        else:
            test_runs()

    # def test_major_all_minor_slice_step(self):
    #
    #     with pytest.raises(NotImplementedError):
    #         # positive step incr
    #         self._run(slice(None), slice(1, 10, 2))
    #         self._run(slice(None), slice(2, 10, 5))
    #         self._run(slice(None), slice(0, 10, 10))
    #
    #         # positive step decr
    #         self._run(slice(None), slice(10, 1, 2))
    #         self._run(slice(None), slice(10, 2, 5))
    #         self._run(slice(None), slice(10, 0, 10))
    #
    #         # positive step incr
    #         self._run(slice(None), slice(10, 1, 2))
    #         self._run(slice(None), slice(10, 2, 5))
    #         self._run(slice(None), slice(10, 0, 10))
    #
    #         # negative step decr
    #         self._run(slice(None), slice(10, 1, -2))
    #         self._run(slice(None), slice(10, 2, -5))

    def test_major_reorder(self):
        def test_runs():
            self._run(slice(None, None, -1))
            self._run(slice(None, None, -2))
            self._run(slice(None, None, -50))

        if self.format == 'csc':
            with pytest.raises(NotImplementedError):
                test_runs()
        else:
            test_runs()

    def test_major_reorder_minor_reorder(self):
        with pytest.raises(NotImplementedError):
            self._run(slice(None, None, -1), slice(None, None, -1))
            self._run(slice(None, None, -3), slice(None, None, -3))

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
