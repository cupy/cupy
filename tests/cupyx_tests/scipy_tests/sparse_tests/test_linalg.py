import unittest

import numpy
import pytest
try:
    import scipy.sparse
    import scipy.sparse.linalg
    import scipy.stats
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy
from cupy import testing
from cupy.testing import condition
from cupyx.scipy import sparse
import cupyx.scipy.sparse.linalg  # NOQA


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
    'ord': [None, -numpy.Inf, -2, -1, 0, 1, 2, 3, numpy.Inf, 'fro'],
    'dtype': [
        numpy.float32,
        numpy.float64,
        numpy.complex64,
        numpy.complex128
    ],
    'axis': [None, (0, 1), (1, -2)],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
@testing.gpu
class TestMatrixNorm(unittest.TestCase):

    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, sp_name='sp',
                                 accept_error=(ValueError,
                                               NotImplementedError))
    def test_matrix_norm(self, xp, sp):
        a = xp.arange(9, dtype=self.dtype) - 4
        b = a.reshape((3, 3))
        b = sp.csr_matrix(b, dtype=self.dtype)
        return sp.linalg.norm(b, ord=self.ord, axis=self.axis)


@testing.parameterize(*testing.product({
    'ord': [None, -numpy.Inf, -2, -1, 0, 1, 2, numpy.Inf, 'fro'],
    'dtype': [
        numpy.float32,
        numpy.float64,
        numpy.complex64,
        numpy.complex128
    ],
    'transpose': [True, False],
    'axis': [0, (1,), (-2,), -1],
})
)
@unittest.skipUnless(scipy_available, 'requires scipy')
@testing.gpu
class TestVectorNorm(unittest.TestCase):
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, sp_name='sp',
                                 accept_error=(ValueError,))
    def test_vector_norm(self, xp, sp):
        a = xp.arange(9, dtype=self.dtype) - 4
        b = a.reshape((3, 3))
        b = sp.csr_matrix(b, dtype=self.dtype)
        if self.transpose:
            b = b.T
        return sp.linalg.norm(b, ord=self.ord, axis=self.axis)

# TODO : TestVsNumpyNorm


@testing.parameterize(*testing.product({
    'which': ['LM', 'LA'],
    'k': [3, 6, 12],
    'return_eigenvectors': [True, False],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestEigsh(unittest.TestCase):
    n = 30
    density = 0.33
    _tol = {'f': 1e-5, 'd': 1e-12}

    def _make_matrix(self, dtype, xp):
        shape = (self.n, self.n)
        a = testing.shaped_random(shape, xp, dtype=dtype)
        mask = testing.shaped_random(shape, xp, dtype='f', scale=1)
        a[mask > self.density] = 0
        a = a * a.conj().T
        return a

    def _test_eigsh(self, a, xp, sp):
        ret = sp.linalg.eigsh(a, k=self.k, which=self.which,
                              return_eigenvectors=self.return_eigenvectors)
        if self.return_eigenvectors:
            w, x = ret
            # Check the residuals to see if eigenvectors are correct.
            ax_xw = a @ x - xp.multiply(x, w.reshape(1, self.k))
            res = xp.linalg.norm(ax_xw) / xp.linalg.norm(w)
            tol = self._tol[numpy.dtype(a.dtype).char.lower()]
            assert(res < tol)
        else:
            w = ret
        return xp.sort(w)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_sparse(self, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        a = sp.csr_matrix(a)
        return self._test_eigsh(a, xp, sp)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5, sp_name='sp')
    def test_dense(self, dtype, xp, sp):
        a = self._make_matrix(dtype, xp)
        return self._test_eigsh(a, xp, sp)

    def test_invalid(self):
        for xp, sp in ((numpy, scipy.sparse), (cupy, sparse)):
            a = xp.diag(xp.ones((self.n, ), dtype='f'))
            with pytest.raises(ValueError):
                sp.linalg.eigsh(xp.ones((2, 1), dtype='f'))
            with pytest.raises(ValueError):
                sp.linalg.eigsh(a, k=0)
        xp, sp = cupy, sparse
        a = xp.diag(xp.ones((self.n, ), dtype='f'))
        with pytest.raises(ValueError):
            sp.linalg.eigsh(xp.ones((1,), dtype='f'))
        with pytest.raises(TypeError):
            sp.linalg.eigsh(xp.ones((2, 2), dtype='i'))
        with pytest.raises(ValueError):
            sp.linalg.eigsh(a, k=self.n)
        with pytest.raises(ValueError):
            sp.linalg.eigsh(a, k=self.k, which='SM')
        with pytest.raises(ValueError):
            sp.linalg.eigsh(a, k=self.k, which='SA')
