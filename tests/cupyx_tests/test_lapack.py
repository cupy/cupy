import unittest

import numpy

import cupy
from cupy import testing
from cupy.testing import attr
from cupyx import lapack


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'n': [3],
    'nrhs': [None, 1, 4],
}))
@attr.gpu
class TestGesv(unittest.TestCase):
    _tol = {'f': 1e-5, 'd': 1e-12}

    def _make_array(self, shape, alpha, beta):
        a = testing.shaped_random(shape, cupy, dtype=self.r_dtype,
                                  scale=alpha) + beta
        return a

    def _make_matrix(self, shape, alpha, beta):
        a = self._make_array(shape, alpha, beta)
        if self.dtype.char in 'FD':
            a = a + 1j * self._make_array(shape, alpha, beta)
        return a

    def setUp(self):
        self.dtype = numpy.dtype(self.dtype)
        if self.dtype.char in 'fF':
            self.r_dtype = numpy.float32
        else:
            self.r_dtype = numpy.float64
        n = self.n
        nrhs = 1 if self.nrhs is None else self.nrhs
        # Diagonally dominant matrix is used as it is stable
        alpha = 2.0 / n
        a = self._make_matrix((n, n), alpha, -alpha / 2)
        diag = cupy.diag(cupy.ones((n,), dtype=self.r_dtype))
        a[diag > 0] = 0
        a += diag
        x = self._make_matrix((n, nrhs), 0.2, 0.9)
        b = cupy.matmul(a, x)
        b_shape = [n]
        if self.nrhs is not None:
            b_shape.append(nrhs)
        self.a = a
        self.b = b.reshape(b_shape)
        self.x_ref = x.reshape(b_shape)
        if self.r_dtype == numpy.float32:
            self.tol = self._tol['f']
        elif self.r_dtype == numpy.float64:
            self.tol = self._tol['d']

    def test_gesv(self):
        x = lapack.gesv(self.a, self.b)
        cupy.testing.assert_allclose(x, self.x_ref,
                                     rtol=self.tol, atol=self.tol)


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
