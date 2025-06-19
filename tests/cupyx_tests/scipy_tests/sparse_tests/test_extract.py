import unittest

import numpy
try:
    import scipy.sparse
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
from cupy import testing
from cupyx.scipy import sparse


@testing.parameterize(*testing.product({
    'shape': [(8, 3), (4, 4), (3, 8)],
    'a_format': ['dense', 'csr', 'csc', 'coo'],
    'out_format': [None, 'csr', 'csc'],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestExtract(unittest.TestCase):

    density = 0.75

    def _make_matrix(self, dtype):
        a = testing.shaped_random(self.shape, numpy, dtype=dtype)
        a[a > self.density] = 0
        b = cupy.array(a)
        if self.a_format == 'csr':
            a = scipy.sparse.csr_matrix(a)
            b = sparse.csr_matrix(b)
        elif self.a_format == 'csc':
            a = scipy.sparse.csc_matrix(a)
            b = sparse.csc_matrix(b)
        elif self.a_format == 'coo':
            a = scipy.sparse.coo_matrix(a)
            b = sparse.coo_matrix(b)
        return a, b

    @testing.for_dtypes('fdFD')
    def test_tril(self, dtype):
        np_a, cp_a = self._make_matrix(dtype)
        m, n = self.shape
        for k in range(-m+1, n):
            np_out = scipy.sparse.tril(np_a, k=k, format=self.out_format)
            cp_out = sparse.tril(cp_a, k=k, format=self.out_format)
            assert np_out.format == cp_out.format
            assert np_out.nnz == cp_out.nnz
            cupy.testing.assert_allclose(np_out.todense(), cp_out.todense())

    @testing.for_dtypes('fdFD')
    def test_triu(self, dtype):
        np_a, cp_a = self._make_matrix(dtype)
        m, n = self.shape
        for k in range(-m+1, n):
            np_out = scipy.sparse.triu(np_a, k=k, format=self.out_format)
            cp_out = sparse.triu(cp_a, k=k, format=self.out_format)
            assert np_out.format == cp_out.format
            assert np_out.nnz == cp_out.nnz
            cupy.testing.assert_allclose(np_out.todense(), cp_out.todense())

    @testing.for_dtypes('fdFD')
    def test_find(self, dtype):
        if self.out_format is not None:
            unittest.SkipTest()
        np_a, cp_a = self._make_matrix(dtype)
        np_row, np_col, np_data = scipy.sparse.find(np_a)
        cp_row, cp_col, cp_data = sparse.find(cp_a)
        # Note: Check the results by reconstructing the sparse matrix from the
        # results of find, as SciPy and CuPy differ in the data order.
        np_out = scipy.sparse.coo_matrix((np_data, (np_row, np_col)),
                                         shape=self.shape)
        cp_out = sparse.coo_matrix((cp_data, (cp_row, cp_col)),
                                   shape=self.shape)
        cupy.testing.assert_allclose(np_out.todense(), cp_out.todense())
