import unittest

import cupy

from cupy import sparse
from cupy import testing

import pytest


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.5],
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n_rows': [15000],
    'n_cols': [15000]
}))
@testing.with_requires('scipy')
class TestSetitemIndexing(unittest.TestCase):

    def _run(self, maj, min=None, data=5):

        for i in range(2):
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

            import time

            if min is not None:
                expected = a.get()

                cupy.cuda.Stream.null.synchronize()
                cpu_time = time.time()
                expected[maj_h, min_h] = data
                cpu_stop = time.time() - cpu_time

                actual = a

                gpu_time = time.time()
                cupy.cuda.Stream.null.synchronize()
                actual[maj, min] = data
                cupy.cuda.Stream.null.synchronize()
                gpu_stop = time.time() - gpu_time
            else:
                expected = a.get()

                cupy.cuda.Stream.null.synchronize()
                cpu_time = time.time()
                expected[maj_h] = data
                cpu_stop = time.time() - cpu_time

                actual = a

                gpu_time = time.time()
                cupy.cuda.Stream.null.synchronize()
                actual[maj] = data
                cupy.cuda.Stream.null.synchronize()
                gpu_stop = time.time() - gpu_time

        print("maj=%s, min=%s, format=%s, cpu_time=%s, gpu_time=%s"
              % (maj, min, self.format, cpu_stop, gpu_stop))

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

    def test_major_scalar_minor_fancy(self):
        self._run(5, [1, 5, 4])

    def test_major_all_minor_all(self):
        self._run(slice(None), slice(None))

    def test_major_all_minor_fancy(self):
        self._run([1, 2, 3, 4, 1, 6, 1, 8, 9], [1, 5, 2, 3, 4, 5, 4, 1, 5])
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

    def test_fancy_setting(self):
        self._run(0, 0, 5)
        self._run([0, 5, 10, 2], 0, [1, 2, 3, 2])

        self._run([[True], [False], [True]], data=5)
        self._run([[True], [False], [False], [True], [True], [True]], data=5)
        self._run([True, False, False, True, True, True], data=5)


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.2, 0.8],
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n_rows': [25, 150],
    'n_cols': [25, 150]
}))
@testing.with_requires('scipy>=1.4.0')
@testing.gpu
class TestGetItemIndexing(unittest.TestCase):

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

        if min is not None:
            expected = expected[maj, min]
            actual = a[maj, min]
        else:
            expected = expected[maj]
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
