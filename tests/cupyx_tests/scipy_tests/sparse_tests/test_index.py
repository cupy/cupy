import unittest

import cupy
from cupy import testing

import pytest


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.9],
    'dtype': ['float32', 'float64'],#, 'complex64', 'complex128'],
    'n_rows': [15],
    'n_cols': [15]
}))
@testing.with_requires('scipy')
class TestSetitemIndexing(unittest.TestCase):

    def _run(self, maj, min=None, data=5):
        a = cupy.sparse.random(self.n_rows, self.n_cols,
                               format=self.format,
                               density=self.density)

        a = a.astype(self.dtype)

        # sparse.random doesn't support complex types
        # so we need to cast
        a = a.astype(self.dtype)

        if isinstance(maj, cupy.ndarray):
            maj_h = maj.get()
        else:
            maj_h = maj

        if isinstance(min, cupy.ndarray):
            min_h = min.get()
        else:
            min_h = min

        if min is not None:
            expected = a.get()
            expected[maj_h, min_h] = data
            actual = a
            actual[maj, min] = data
            cupy.cuda.Stream.null.synchronize()
        else:
            expected = a.get()

            expected[maj_h] = data
            actual = a

            actual[maj] = data
            cupy.cuda.Stream.null.synchronize()

        if cupy.sparse.isspmatrix(actual):
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

    def test_major_slice(self):
        self._run(slice(5, 9))
        self._run(slice(9, 5))
        self._run(slice(5, 9), data=[5, 6, 7, 8])
        self._run(slice(5, 9, 2), data=[5, 7])

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

    def test_major_scalar_minor_fancy(self):
        self._run(5, [1, 5, 4])

    def test_major_all_minor_all(self):
        self._run(slice(None), slice(None))

    def test_major_all_minor_fancy(self):
        self._run(slice(None), [1, 5, 2, 3, 4, 5, 4, 1, 5])
        self._run(slice(None), [0, 3, 4, 1, 1, 5, 5, 2, 3, 4, 5, 4, 1, 5])

    def test_major_fancy_minor_fancy(self):
        self._run([1, 5, 4], [1, 5, 4])
        self._run([2, 0, 10], [9, 2, 1])
        self._run([2, 0], [2, 1])

    def test_major_fancy_minor_all(self):
        self._run([1, 5, 4], slice(None))

    def test_major_fancy_minor_scalar(self):
        self._run([1, 5, 4], 5)

    def test_major_fancy_minor_slice(self):
        self._run([1, 5, 4], slice(1, 5))
        self._run([1, 5, 4], slice(5, 1, -1))

    def test_major_bool_fancy(self):
        rand_bool = cupy.random.random(self.n_rows).astype(cupy.bool)
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

    def test_major_slice(self):
        self._run(slice(5, 10000), data=5)
        self._run(slice(5, 4), data=5)
        self._run(slice(4, 5, 2), data=5)
        self._run(slice(5, 4, -2), data=5)

        self._run(slice(2, 4), slice(0, 2), [[4], [1]])
        self._run(slice(2, 4), slice(0, 2), [[4, 5], [6, 7]])
        self._run(slice(2, 4), 0, [[4], [6]])

        self._run(0, 0, 5)
        self._run([0, 5, 10, 2], 0, [1, 2, 3, 2])

        self._run([[True], [False], [True]], data=5)
        self._run([[True], [False], [False], [True], [True], [True]], data=5)
        self._run([True, False, False, True, True, True], data=5)


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.1, 0.5, 0.9],
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n_rows': [25, 100],
    'n_cols': [25, 100]
}))
@testing.with_requires('scipy')
class TestIndexing(unittest.TestCase):

    def _run(self, maj, min=None):
        a = cupy.sparse.random(self.n_rows, self.n_cols,
                               format=self.format,
                               density=self.density)

        # sparse.random doesn't support complex types
        # so we need to cast
        a = a.astype(self.dtype)

        if isinstance(maj, cupy.ndarray):
            maj_h = maj.get()
        else:
            maj_h = maj

        if isinstance(min, cupy.ndarray):
            min_h = min.get()
        else:
            min_h = min

        if min is not None:
            expected = a.get()[maj_h, min_h]
            actual = a[maj, min]
            cupy.cuda.Stream.null.synchronize()
        else:
            expected = a.get()[maj_h]
            actual = a[maj]
            cupy.cuda.Stream.null.synchronize()

        if cupy.sparse.isspmatrix(actual):
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

    def test_major_slice(self):
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

    def test_major_scalar_minor_fancy(self):
        self._run(5, [1, 5, 4])

    def test_major_all_minor_scalar(self):
        self._run(slice(None), 5)

    def test_major_all_minor_slice(self):
        self._run(slice(None), slice(5, 10))

    def test_major_all_minor_all(self):
        self._run(slice(None), slice(None))

    def test_major_all_minor_fancy(self):
        self._run(slice(None), [1, 5, 2, 3, 4, 5, 4, 1, 5])
        self._run(slice(None), [0, 3, 4, 1, 1, 5, 5, 2, 3, 4, 5, 4, 1, 5])

    def test_major_fancy_minor_fancy(self):
        self._run([1, 5, 4], [1, 5, 4])
        self._run([2, 0, 10], [9, 2, 1])
        self._run([2, 0], [2, 1])

    def test_major_fancy_minor_all(self):
        self._run([1, 5, 4], slice(None))

    def test_major_fancy_minor_scalar(self):
        self._run([1, 5, 4], 5)

    def test_major_fancy_minor_slice(self):
        self._run([1, 5, 4], slice(1, 5))
        self._run([1, 5, 4], slice(5, 1, -1))

    def test_major_bool_fancy(self):
        rand_bool = cupy.random.random(self.n_rows).astype(cupy.bool)
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

    def test_major_all_minor_slice_step(self):

        # positive step incr
        self._run(slice(None), slice(1, 10, 2))
        self._run(slice(None), slice(2, 10, 5))
        self._run(slice(None), slice(0, 10, 10))

        # positive step decr
        self._run(slice(None), slice(10, 1, 2))
        self._run(slice(None), slice(10, 2, 5))
        self._run(slice(None), slice(10, 0, 10))

        # positive step incr
        self._run(slice(None), slice(10, 1, 2))
        self._run(slice(None), slice(10, 2, 5))
        self._run(slice(None), slice(10, 0, 10))

        # negative step decr
        self._run(slice(None), slice(10, 1, -2))
        self._run(slice(None), slice(10, 2, -5))
        self._run(slice(None), slice(10, 0, -10))

    def test_major_reorder(self):
        self._run(slice(None, None, -1))
        self._run(slice(None, None, -2))
        self._run(slice(None, None, -50))

    def test_major_reorder_minor_reorder(self):
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

@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.1, 0.5, 0.9],
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n_rows': [25, 100],
    'n_cols': [25, 100]
}))
@testing.with_requires('scipy')
class TestIndexing(unittest.TestCase):

    def _run(self, maj, min=None):
        a = cupy.sparse.random(self.n_rows, self.n_cols,
                               format=self.format,
                               density=self.density)

        # sparse.random doesn't support complex types
        # so we need to cast
        a = a.astype(self.dtype)

        if isinstance(maj, cupy.ndarray):
            maj_h = maj.get()
        else:
            maj_h = maj

        if isinstance(min, cupy.ndarray):
            min_h = min.get()
        else:
            min_h = min

        if min is not None:
            expected = a.get()[maj_h, min_h]
            actual = a[maj, min]
            cupy.cuda.Stream.null.synchronize()
        else:
            expected = a.get()[maj_h]
            actual = a[maj]
            cupy.cuda.Stream.null.synchronize()

        if cupy.sparse.isspmatrix(actual):
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

    def test_major_slice(self):
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

    def test_major_scalar_minor_fancy(self):
        self._run(5, [1, 5, 4])

    def test_major_all_minor_scalar(self):
        self._run(slice(None), 5)

    def test_major_all_minor_slice(self):
        self._run(slice(None), slice(5, 10))

    def test_major_all_minor_all(self):
        self._run(slice(None), slice(None))

    def test_major_all_minor_fancy(self):
        self._run(slice(None), [1, 5, 2, 3, 4, 5, 4, 1, 5])
        self._run(slice(None), [0, 3, 4, 1, 1, 5, 5, 2, 3, 4, 5, 4, 1, 5])

    def test_major_fancy_minor_fancy(self):
        self._run([1, 5, 4], [1, 5, 4])
        self._run([2, 0, 10], [9, 2, 1])
        self._run([2, 0], [2, 1])

    def test_major_fancy_minor_all(self):
        self._run([1, 5, 4], slice(None))

    def test_major_fancy_minor_scalar(self):
        self._run([1, 5, 4], 5)

    def test_major_fancy_minor_slice(self):
        self._run([1, 5, 4], slice(1, 5))
        self._run([1, 5, 4], slice(5, 1, -1))

    def test_major_bool_fancy(self):
        rand_bool = cupy.random.random(self.n_rows).astype(cupy.bool)
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

    def test_major_all_minor_slice_step(self):

        # positive step incr
        self._run(slice(None), slice(1, 10, 2))
        self._run(slice(None), slice(2, 10, 5))
        self._run(slice(None), slice(0, 10, 10))

        # positive step decr
        self._run(slice(None), slice(10, 1, 2))
        self._run(slice(None), slice(10, 2, 5))
        self._run(slice(None), slice(10, 0, 10))

        # positive step incr
        self._run(slice(None), slice(10, 1, 2))
        self._run(slice(None), slice(10, 2, 5))
        self._run(slice(None), slice(10, 0, 10))

        # negative step decr
        self._run(slice(None), slice(10, 1, -2))
        self._run(slice(None), slice(10, 2, -5))
        self._run(slice(None), slice(10, 0, -10))

    def test_major_reorder(self):
        self._run(slice(None, None, -1))
        self._run(slice(None, None, -2))
        self._run(slice(None, None, -50))

    def test_major_reorder_minor_reorder(self):
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