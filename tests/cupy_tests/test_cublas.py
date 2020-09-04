import unittest

import numpy

import cupy
from cupy import cublas
from cupy import testing
from cupy.testing import attr


@testing.parameterize(*testing.product({
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n': [10, 33, 100],
    'bs': [None, 1, 100],
    'nrhs': [None, 1, 10],
}))
@attr.gpu
class TestBatchedGesv(unittest.TestCase):
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
        bs = 1 if self.bs is None else self.bs
        nrhs = 1 if self.nrhs is None else self.nrhs
        # Diagonally dominant matrix is used as it is stable
        alpha = 2.0 / n
        a = self._make_matrix((bs, n, n), alpha, -alpha / 2)
        diag = cupy.diag(cupy.ones((n,), dtype=self.r_dtype))
        for i in range(bs):
            a[i][diag > 0] = 0
            a[i] += diag
        x = self._make_matrix((bs, n, nrhs), 0.2, 0.9)
        b = cupy.matmul(a, x)
        a_shape = (n, n) if self.bs is None else (bs, n, n)
        b_shape = [n]
        if self.bs is not None:
            b_shape.insert(0, bs)
        if self.nrhs is not None:
            b_shape.append(nrhs)
        self.a = a.reshape(a_shape)
        self.b = b.reshape(b_shape)
        self.x_ref = x.reshape(b_shape)
        if self.r_dtype == numpy.float32:
            self.tol = self._tol['f']
        elif self.r_dtype == numpy.float64:
            self.tol = self._tol['d']

    def test_batched_gesv(self):
        x = cublas.batched_gesv(self.a, self.b)
        cupy.testing.assert_allclose(x, self.x_ref,
                                     rtol=self.tol, atol=self.tol)
