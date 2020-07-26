import unittest

import numpy
import pytest
try:
    import scipy.sparse
    import scipy.stats
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
from cupy import testing
from cupy.testing import condition
from cupyx.scipy import sparse


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestLsqr(unittest.TestCase):

    def setUp(self):
        rvs = scipy.stats.randint(0, 15).rvs
        self.A = scipy.sparse.random(50, 50, density=0.2, data_rvs=rvs)
        self.b = numpy.random.randint(15, size=50)

    def test_size(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            A = sp.csr_matrix(self.A, dtype=self.dtype)
            b = xp.array(numpy.append(self.b, [1]), dtype=self.dtype)
            with pytest.raises(ValueError):
                sp.linalg.lsqr(A, b)

    def test_shape(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            A = sp.csr_matrix(self.A, dtype=self.dtype)
            b = xp.array(numpy.tile(self.b, (2, 1)), dtype=self.dtype)
            with pytest.raises(ValueError):
                sp.linalg.lsqr(A, b)

    @condition.retry(10)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_csrmatrix(self, xp, sp):
        A = sp.csr_matrix(self.A, dtype=self.dtype)
        b = xp.array(self.b, dtype=self.dtype)
        x = sp.linalg.lsqr(A, b)
        return x[0]

    @condition.retry(10)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_ndarray(self, xp, sp):
        A = xp.array(self.A.A, dtype=self.dtype)
        b = xp.array(self.b, dtype=self.dtype)
        x = sp.linalg.lsqr(A, b)
        return x[0]


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestSpilu(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(0)
        N = 40
        data = numpy.ones((3, N))
        data[0, :] = 2
        data[1, :] = -1
        data[2, :] = -1
        self.A_poisson = scipy.sparse.spdiags(data, [0, -1, 1], N, N,
                                              format='csr')
        self.A_poisson.sort_indices()
        self.b_poisson = numpy.random.randn(N)

        N = 5
        self.A_rand = numpy.randn((N, N))
        self.A_rand.sort_indices()
        self.b_rand = numpy.random.randn(N)

    def test_size(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            A = sp.csr_matrix(self.A, dtype=self.dtype)
            b = xp.array(numpy.append(self.b, [1]), dtype=self.dtype)
            with pytest.raises(ValueError):
                if sp == scipy.sparse:
                    sp.linalg.spilu(A).solve(b)
                else:
                    sp.linalg.spilu(A)(b)

    def test_shape(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            A = sp.csr_matrix(self.A, dtype=self.dtype)
            b = xp.array(numpy.tile(self.b, (2, 1)), dtype=self.dtype)
            with pytest.raises(ValueError):
                if sp == scipy.sparse:
                    sp.linalg.spilu(A).solve(b)
                else:
                    sp.linalg.spilu(A)(b)

    @condition.retry(10)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_csrmatrix_poisson(self, xp, sp):
        A = sp.csr_matrix(self.A_poisson, dtype=self.dtype)
        b = xp.array(self.b_poisson, dtype=self.dtype)
        if sp == scipy.sparse:
            x = sp.linalg.spilu(A).solve(b)
        else:
            x = sp.linalg.spilu(A)(b)
        return x[0]

    @condition.retry(10)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_csrmatrix_rand(self, xp, sp):
        A = sp.csr_matrix(self.A_rand, dtype=self.dtype)
        b = xp.array(self.b_rand, dtype=self.dtype)
        if sp == scipy.sparse:
            x = sp.linalg.spilu(A).solve(b)
        else:
            x = sp.linalg.spilu(A)(b)
        return x[0]

    @condition.retry(10)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_ndarray_poisson(self, xp, sp):
        A = xp.array(self.A_poisson.A, dtype=self.dtype)
        b = xp.array(self.b_poisson, dtype=self.dtype)
        if sp == scipy.sparse:
            x = sp.linalg.spilu(A).solve(b)
        else:
            x = sp.linalg.spilu(A)(b)
        return x[0]

    @condition.retry(10)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_ndarray_rand(self, xp, sp):
        A = xp.array(self.A_rand.A, dtype=self.dtype)
        b = xp.array(self.b_rand, dtype=self.dtype)
        if sp == scipy.sparse:
            x = sp.linalg.spilu(A).solve(b)
        else:
            x = sp.linalg.spilu(A)(b)
        return x[0]


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64,
              numpy.complex128],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestBicgstab(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(0)
        N = 40
        data = numpy.ones((3, N))
        data[0, :] = 2
        data[1, :] = -1
        data[2, :] = -1
        self.A_poisson = scipy.sparse.spdiags(data, [0, -1, 1], N, N,
                                              format='csr')
        self.A_poisson.sort_indices()
        self.b_poisson = numpy.random.randn(N)

        N = 5
        self.A_rand = numpy.randn((N, N))
        self.A_rand.sort_indices()
        self.b_rand = numpy.random.randn(N)

    @condition.retry(10)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_csrmatrix_poisson(self, xp, sp):
        A = sp.csr_matrix(self.A_poisson, dtype=self.dtype)
        b = xp.array(self.b_poisson, dtype=self.dtype)
        x = sp.linalg.bicgstab(A, b, atol=0, tol=1e-7)
        return x[0]

    @condition.retry(10)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_csrmatrix_rand(self, xp, sp):
        A = sp.csr_matrix(self.A_rand, dtype=self.dtype)
        b = xp.array(self.b_rand, dtype=self.dtype)
        x = sp.linalg.bicgstab(A, b, atol=0, tol=1e-7)
        return x[0]

    @condition.retry(10)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_ndarray_poisson(self, xp, sp):
        A = xp.array(self.A_poisson.A, dtype=self.dtype)
        b = xp.array(self.b_poisson, dtype=self.dtype)
        x = sp.linalg.bicgstab(A, b, atol=0)
        return x[0]

    @condition.retry(10)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_ndarray_rand(self, xp, sp):
        A = xp.array(self.A_rand.A, dtype=self.dtype)
        b = xp.array(self.b_rand, dtype=self.dtype)
        x = sp.linalg.bicgstab(A, b, atol=0)
        return x[0]
