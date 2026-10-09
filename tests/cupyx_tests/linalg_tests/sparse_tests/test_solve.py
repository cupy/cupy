from __future__ import annotations

import unittest

import numpy
import pytest
try:
    import scipy.linalg
    import scipy.sparse
    import scipy.stats

    scipy_available = True
except ImportError:
    scipy_available = False

import cupy as cp
from cupy import testing
from cupy.testing import _condition
import cupyx
import cupyx.scipy.sparse as sp


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestLschol(unittest.TestCase):

    def setUp(self):
        rvs = scipy.stats.randint(0, 15).rvs
        self.A = scipy.sparse.random(
            50, 50, density=0.2, data_rvs=rvs, dtype=self.dtype)
        self.b = numpy.random.randint(5, size=50)
        # symmetric and positive definite
        self.A = self.A.T*self.A + 10*scipy.sparse.eye(50)
        self.b = self.A.T*self.b
        # initial scipy results by dense cholesky method.
        L = scipy.linalg.cho_factor(self.A.todense())
        self.x = scipy.linalg.cho_solve(L, self.b)
        if self.dtype == numpy.float64:
            self.decimal = 8
        else:
            self.decimal = 3

    def test_size(self):
        A = sp.csr_matrix(self.A, dtype=self.dtype)
        b = cp.array(numpy.append(self.b, [1]), dtype=self.dtype)
        with pytest.raises(ValueError):
            cupyx.linalg.sparse.lschol(A, b)

    def test_shape(self):
        A = sp.csr_matrix(self.A, dtype=self.dtype)
        b = cp.array(numpy.tile(self.b, (2, 1)), dtype=self.dtype)
        with pytest.raises(ValueError):
            cupyx.linalg.sparse.lschol(A, b)

    @_condition.retry(10)
    def test_csrmatrix(self):
        A = sp.csr_matrix(self.A, dtype=self.dtype)
        b = cp.array(self.b, dtype=self.dtype)
        x = cupyx.linalg.sparse.lschol(A, b)
        testing.assert_array_almost_equal(x, self.x, decimal=self.decimal)

    @_condition.retry(10)
    def test_ndarray(self):
        A = cp.array(self.A.toarray(), dtype=self.dtype)
        b = cp.array(self.b, dtype=self.dtype)
        x = cupyx.linalg.sparse.lschol(A, b)
        testing.assert_array_almost_equal(x, self.x, decimal=self.decimal)

    def test_strided_rhs(self):
        A = sp.csr_matrix(numpy.diag([2.0, 3.0]), dtype=self.dtype)
        b = cp.array([2.0, 0.0, 6.0, 0.0], dtype=self.dtype)[::2]
        assert not b.flags.c_contiguous
        x = cupyx.linalg.sparse.lschol(A, b)
        testing.assert_array_almost_equal(x, [1.0, 2.0], decimal=self.decimal)


@pytest.mark.parametrize('a_dtype,b_dtype', [
    (cp.complex64, cp.float32),
    (cp.float32, cp.complex128),
])
def test_lschol_rejects_complex(a_dtype, b_dtype):
    A = sp.eye(2, format='csr', dtype=a_dtype)
    b = cp.ones(2, dtype=b_dtype)
    with pytest.raises(TypeError, match='Invalid dtype'):
        cupyx.linalg.sparse.lschol(A, b)
