import unittest

from cupy import sparse
from cupy import testing
import cupy

import pytest


@testing.parameterize(*testing.product({
    'format': ['csr', 'csc'],
    'density': [0.5],
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n_rows': [15000],
    'n_cols': [15000]
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

        for i in range(2):

            expected = a.get()

            import time
            if min is not None:

                cpu_time = time.time()
                expected = expected[maj, min]
                cpu_end = time.time() - cpu_time

                gpu_time = time.time()
                actual = a[maj, min]
                cupy.cuda.Stream.null.synchronize()
                gpu_end = time.time() - gpu_time

                d = a.toarray()
                cupy.cuda.Stream.null.synchronize()
                gpu_dense_time = time.time()
                d[maj, min]
                cupy.cuda.Stream.null.synchronize()
                gpu_dense_end = time.time() - gpu_dense_time
            else:
                cpu_time = time.time()
                expected = expected[maj]
                cpu_end = time.time() - cpu_time

                gpu_time = time.time()
                actual = a[maj]
                cupy.cuda.Stream.null.synchronize()
                gpu_end = time.time() - gpu_time

                d = a.toarray()
                cupy.cuda.Stream.null.synchronize()
                gpu_dense_time = time.time()
                d[maj]
                cupy.cuda.Stream.null.synchronize()
                gpu_dense_end = time.time() - gpu_dense_time

        print("format=%s, maj=%s, min=%s, dtype=%s, rows=%s, cols=%s, density=%s, cpu_time=%s, gpu_time=%s, gpu_dense_time=%s" %
              (self.format, maj, min, self.dtype, self.n_rows, self.n_cols, self.density, cpu_end, gpu_end, gpu_dense_end))

        # if sparse.isspmatrix(actual):
        #     actual.sort_indices()
        #     expected.sort_indices()
        #
        #     testing.assert_array_equal(
        #         actual.indptr, expected.indptr)
        #     testing.assert_array_equal(
        #         actual.indices, expected.indices)
        #     testing.assert_array_equal(
        #         actual.data, expected.data)
        # else:
        #     testing.assert_array_equal(
        #         actual, expected)

    def test_major_slice(self):
        self._run(slice(5, 900))
        self._run(slice(9, 500))

    def test_major_all(self):
        self._run(slice(None))

    def test_major_scalar(self):
        self._run(10)

    def test_major_slice_minor_slice(self):
        self._run(slice(1, 500), slice(1, 500))

    def test_major_slice_minor_all(self):
        self._run(slice(1, 500), slice(None))
        self._run(slice(500, 1), slice(None))

    def test_major_scalar_minor_slice(self):
        self._run(5, slice(1, 500))

    def test_major_scalar_minor_all(self):
        self._run(5, slice(None))

    def test_major_scalar_minor_scalar(self):
        self._run(5, 5)

    def test_major_all_minor_scalar(self):
        self._run(slice(None), 5)

    def test_major_all_minor_slice(self):
        self._run(slice(None), slice(5, 1000))

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
