import unittest

import numpy
try:
    import scipy.sparse
    import scipy.stats
    import scipy.linalg
    scipy_available = True
except ImportError:
    scipy_available = False

from cupy import cuda
from cupy import testing
from cupy.testing import condition
import cupy.sparse as sp
import cupy as cp
import cupyx


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@unittest.skipUnless(
    cuda.cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestLschol(unittest.TestCase):

    def setUp(self):
        rvs = scipy.stats.randint(0, 15).rvs
        self.A = scipy.sparse.random(
            50, 50, density=0.2, data_rvs=rvs, dtype=self.dtype)
        self.b = numpy.random.randint(15, size=50)
        self.A = self.A.T*self.A  # symmetric and positive definite
        self.b = self.A.T*self.b
        # inital scipy results by dense cholesky method.
        L = scipy.linalg.cho_factor(self.A.todense())
        self.x = scipy.linalg.cho_solve(L, self.b)
        if self.dtype == numpy.float64:
            self.decimal = 6
        else:
            self.decimal = 3

    @testing.numpy_cupy_raises()
    def test_size(self):
        A = sp.csr_matrix(self.A, dtype=self.dtype)
        b = cp.array(numpy.append(self.b, [1]), dtype=self.dtype)
        cupyx.linalg.sparse.lschol(A, b)

    @testing.numpy_cupy_raises()
    def test_shape(self):
        A = sp.csr_matrix(self.A, dtype=self.dtype)
        b = cp.array(numpy.tile(self.b, (2, 1)), dtype=self.dtype)
        cupyx.linalg.sparse.lschol(A, b)

    @condition.retry(10)
    def test_csrmatrix(self):
        A = sp.csr_matrix(self.A, dtype=self.dtype)
        b = cp.array(self.b, dtype=self.dtype)
        x = cupyx.linalg.sparse.lschol(A, b)
        testing.assert_array_almost_equal(x, self.x, decimal=self.decimal)

    @condition.retry(10)
    def test_ndarray(self):
        A = cp.array(self.A.A, dtype=self.dtype)
        b = cp.array(self.b, dtype=self.dtype)
        x = cupyx.linalg.sparse.lschol(A, b)
        testing.assert_array_almost_equal(x, self.x, decimal=self.decimal)
