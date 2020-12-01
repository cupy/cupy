import unittest

import numpy
import pytest

import cupy
from cupy import testing
from cupy.testing import attr
from cupyx import lapack


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'n': [3],
    'nrhs': [None, 1, 4],
    'order': ['C', 'F'],
}))
@attr.gpu
class TestGesv(unittest.TestCase):
    _tol = {'f': 1e-5, 'd': 1e-12}

    def _make_array(self, shape, alpha, beta):
        a = testing.shaped_random(shape, cupy, dtype=self.dtype.char.lower(),
                                  order=self.order, scale=alpha) + beta
        return a

    def _make_matrix(self, shape, alpha, beta):
        a = self._make_array(shape, alpha, beta)
        if self.dtype.char in 'FD':
            a = a + 1j * self._make_array(shape, alpha, beta)
        return a

    def setUp(self):
        self.dtype = numpy.dtype(self.dtype)
        n = self.n
        nrhs = 1 if self.nrhs is None else self.nrhs
        # Diagonally dominant matrix is used as it is stable
        alpha = 2.0 / n
        a = self._make_matrix((n, n), alpha, -alpha / 2)
        diag = cupy.diag(cupy.ones((n,), dtype=self.dtype.char.lower()))
        a[diag > 0] = 0
        a += diag
        x = self._make_matrix((n, nrhs), 0.2, 0.9)
        b = cupy.matmul(a, x)
        b_shape = [n]
        if self.nrhs is not None:
            b_shape.append(nrhs)
        b = b.reshape(b_shape)
        self.a = a
        if self.nrhs is None or self.nrhs == 1:
            self.b = b.copy(order=self.order)
        else:
            self.b = b.copy(order='F')
        self.x_ref = x.reshape(b_shape)
        self.tol = self._tol[self.dtype.char.lower()]

    def test_gesv(self):
        lapack.gesv(self.a, self.b)
        cupy.testing.assert_allclose(self.b, self.x_ref,
                                     rtol=self.tol, atol=self.tol)

    def test_invalid_cases(self):
        if self.nrhs is None or self.nrhs == 1:
            raise unittest.SkipTest()
        ng_a = self.a.reshape(1, self.n, self.n)
        with pytest.raises(ValueError):
            lapack.gesv(ng_a, self.b)
        ng_b = self.b.reshape(1, self.n, self.nrhs)
        with pytest.raises(ValueError):
            lapack.gesv(self.a, ng_b)
        ng_a = cupy.ones((self.n, self.n+1), dtype=self.dtype)
        with pytest.raises(ValueError):
            lapack.gesv(ng_a, self.b)
        ng_a = cupy.ones((self.n+1, self.n+1), dtype=self.dtype)
        with pytest.raises(ValueError):
            lapack.gesv(ng_a, self.b)
        ng_a = cupy.ones(self.a.shape, dtype='i')
        with pytest.raises(TypeError):
            lapack.gesv(ng_a, self.b)
        ng_a = cupy.ones((2, self.n, self.n), dtype=self.dtype, order='F')[0]
        with pytest.raises(ValueError):
            lapack.gesv(ng_a, self.b)
        ng_b = self.b.copy(order='C')
        with pytest.raises(ValueError):
            lapack.gesv(self.a, ng_b)


@testing.parameterize(*testing.product({
    'shape': [(4, 4), (5, 4), (4, 5)],
    'nrhs': [None, 1, 4],
}))
@attr.gpu
class TestGels(unittest.TestCase):
    _tol = {'f': 1e-5, 'd': 1e-12}

    @testing.for_dtypes('fdFD')
    def test_gels(self, dtype):
        b_shape = [self.shape[0]]
        if self.nrhs is not None:
            b_shape.append(self.nrhs)
        a = testing.shaped_random(self.shape, numpy, dtype=dtype)
        b = testing.shaped_random(b_shape, numpy, dtype=dtype)
        tol = self._tol[numpy.dtype(dtype).char.lower()]
        x_lstsq = numpy.linalg.lstsq(a, b)[0]
        x_gels = lapack.gels(cupy.array(a), cupy.array(b))
        cupy.testing.assert_allclose(x_gels, x_lstsq, rtol=tol, atol=tol)
