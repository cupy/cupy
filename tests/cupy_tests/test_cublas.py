import unittest

import numpy

import cupy
from cupy import cublas
from cupy import testing
from cupy.testing import attr


@testing.parameterize(*testing.product({
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n': [10, 33, 100],
    'bs': [None, 1, 10],
    'nrhs': [None, 1, 10],
}))
@attr.gpu
class TestBatchedGesv(unittest.TestCase):
    _tol = {'f': 5e-5, 'd': 1e-12}

    def _make_random_matrices(self, shape, xp):
        a = testing.shaped_random(shape, xp, dtype=self.r_dtype, scale=1)
        if self.dtype.char in 'FD':
            a = a + 1j * testing.shaped_random(shape, xp, dtype=self.r_dtype,
                                               scale=1)
        return a

    def _make_well_conditioned_matrices(self, shape):
        a = self._make_random_matrices(shape, numpy)
        u, s, vh = numpy.linalg.svd(a)
        s = testing.shaped_random(s.shape, numpy, dtype=self.r_dtype,
                                  scale=1) + 1
        a = numpy.einsum('...ik,...k,...kj->...ij', u, s, vh)
        return cupy.array(a)

    def setUp(self):
        self.dtype = numpy.dtype(self.dtype)
        if self.dtype.char in 'fF':
            self.r_dtype = numpy.float32
        else:
            self.r_dtype = numpy.float64
        n = self.n
        bs = 1 if self.bs is None else self.bs
        nrhs = 1 if self.nrhs is None else self.nrhs
        a = self._make_well_conditioned_matrices((bs, n, n))
        x = self._make_random_matrices((bs, n, nrhs), cupy)
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


@testing.parameterize(*testing.product({
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'n': [10, 100],
    'mode': [None, numpy, cupy],
}))
@attr.gpu
class TestLevel1Functions(unittest.TestCase):
    _tol = {'f': 1e-5, 'd': 1e-12}

    def setUp(self):
        self.dtype = numpy.dtype(self.dtype)
        self.tol = self._tol[self.dtype.char.lower()]

    def _make_random_vector(self):
        return testing.shaped_random((self.n,), cupy, dtype=self.dtype)

    def _make_out(self, dtype):
        out = None
        if self.mode is not None:
            out = self.mode.empty([], dtype=dtype)
        return out

    def _check_pointer(self, a, b):
        if a is not None and b is not None:
            assert self._get_pointer(a) == self._get_pointer(b)

    def _get_pointer(self, a):
        if isinstance(a, cupy.ndarray):
            return a.data.ptr
        else:
            return a.ctypes.data

    def test_iamax(self):
        x = self._make_random_vector()
        ref = cupy.argmax(cupy.absolute(x.real) + cupy.absolute(x.imag))
        out = self._make_out('i')
        res = cublas.iamax(x, out=out)
        self._check_pointer(res, out)
        # Note: iamax returns 1-based index
        cupy.testing.assert_array_equal(res - 1, ref)

    def test_iamin(self):
        x = self._make_random_vector()
        ref = cupy.argmin(cupy.absolute(x.real) + cupy.absolute(x.imag))
        out = self._make_out('i')
        res = cublas.iamin(x, out=out)
        self._check_pointer(res, out)
        # Note: iamin returns 1-based index
        cupy.testing.assert_array_equal(res - 1, ref)

    def test_asum(self):
        x = self._make_random_vector()
        ref = cupy.sum(cupy.absolute(x.real) + cupy.absolute(x.imag))
        out = self._make_out(self.dtype.char.lower())
        res = cublas.asum(x, out=out)
        self._check_pointer(res, out)
        cupy.testing.assert_allclose(res, ref, rtol=self.tol, atol=self.tol)

    def test_axpy(self):
        x = self._make_random_vector()
        y = self._make_random_vector()
        a = 1.1
        if self.dtype.char in 'FD':
            a = a - 1j * 0.9
        ref = a * x + y
        if self.mode is not None:
            a = self.mode.array(a, dtype=self.dtype)
        cublas.axpy(a, x, y)
        cupy.testing.assert_allclose(y, ref, rtol=self.tol, atol=self.tol)

    def test_dot(self):
        x = self._make_random_vector()
        y = self._make_random_vector()
        ref = x.dot(y)
        out = self._make_out(self.dtype)
        if self.dtype.char in 'FD':
            with self.assertRaises(TypeError):
                res = cublas.dot(x, y, out=out)
            return
        res = cublas.dot(x, y, out=out)
        self._check_pointer(res, out)
        cupy.testing.assert_allclose(res, ref, rtol=self.tol, atol=self.tol)

    def test_dotu(self):
        x = self._make_random_vector()
        y = self._make_random_vector()
        ref = x.dot(y)
        out = self._make_out(self.dtype)
        res = cublas.dotu(x, y, out=out)
        self._check_pointer(res, out)
        cupy.testing.assert_allclose(res, ref, rtol=self.tol, atol=self.tol)

    def test_dotc(self):
        x = self._make_random_vector()
        y = self._make_random_vector()
        ref = x.conj().dot(y)
        out = self._make_out(self.dtype)
        res = cublas.dotc(x, y, out=out)
        self._check_pointer(res, out)
        cupy.testing.assert_allclose(res, ref, rtol=self.tol, atol=self.tol)

    def test_nrm2(self):
        x = self._make_random_vector()
        ref = cupy.linalg.norm(x)
        out = self._make_out(self.dtype.char.lower())
        res = cublas.nrm2(x, out=out)
        self._check_pointer(res, out)
        cupy.testing.assert_allclose(res, ref, rtol=self.tol, atol=self.tol)

    def test_scal(self):
        x = self._make_random_vector()
        a = 1.1
        if self.dtype.char in 'FD':
            a = a - 1j * 0.9
        ref = a * x
        if self.mode is not None:
            a = self.mode.array(a, dtype=self.dtype)
        cublas.scal(a, x)
        cupy.testing.assert_allclose(x, ref, rtol=self.tol, atol=self.tol)


@testing.parameterize(*testing.product({
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'shape': [(10, 9), (9, 10)],
    'trans': ['N', 'T', 'H'],
    'order': ['C', 'F'],
    'mode': [None, numpy, cupy],
}))
@attr.gpu
class TestGemv(unittest.TestCase):
    _tol = {'f': 1e-5, 'd': 1e-12}

    def setUp(self):
        self.dtype = numpy.dtype(self.dtype)
        self.tol = self._tol[self.dtype.char.lower()]

    def test_gemv(self):
        a = testing.shaped_random(self.shape, cupy, dtype=self.dtype,
                                  order=self.order)
        if self.trans == 'N':
            ylen, xlen = self.shape
        else:
            xlen, ylen = self.shape
        x = testing.shaped_random((xlen,), cupy, dtype=self.dtype)
        y = testing.shaped_random((ylen,), cupy, dtype=self.dtype)
        alpha = 0.9
        beta = 0.8
        if self.dtype.char in 'FD':
            alpha = alpha - 1j * 0.7
            beta = beta - 1j * 0.6
        if self.trans == 'N':
            ref = alpha * a.dot(x) + beta * y
        elif self.trans == 'T':
            ref = alpha * a.T.dot(x) + beta * y
        elif self.trans == 'H':
            ref = alpha * a.T.conj().dot(x) + beta * y
        if self.mode is not None:
            alpha = self.mode.array(alpha)
            beta = self.mode.array(beta)
        cupy.cublas.gemv(self.trans, alpha, a, x, beta, y)
        cupy.testing.assert_allclose(y, ref, rtol=self.tol, atol=self.tol)


@testing.parameterize(*testing.product({
    'dtype': ['float32', 'float64', 'complex64', 'complex128'],
    'shape': [(10, 9), (9, 10)],
    'order': ['C', 'F'],
    'mode': [None, numpy, cupy],
}))
@attr.gpu
class TestGer(unittest.TestCase):
    _tol = {'f': 1e-5, 'd': 1e-12}

    def setUp(self):
        self.dtype = numpy.dtype(self.dtype)
        self.tol = self._tol[self.dtype.char.lower()]
        self.a = testing.shaped_random(self.shape, cupy, dtype=self.dtype,
                                       order=self.order)
        self.x = testing.shaped_random((self.shape[0],), cupy,
                                       dtype=self.dtype)
        self.y = testing.shaped_random((self.shape[1],), cupy,
                                       dtype=self.dtype)
        self.alpha = 1.1
        if self.dtype.char in 'FD':
            self.alpha = self.alpha - 1j * 0.9

    def test_ger(self):
        if self.dtype.char in 'FD':
            with self.assertRaises(TypeError):
                cublas.ger(self.alpha, self.x, self.y, self.a)
            return
        ref = self.alpha * cupy.outer(self.x, self.y) + self.a
        if self.mode is not None:
            self.alpha = self.mode.array(self.alpha)
        cublas.ger(self.alpha, self.x, self.y, self.a)
        cupy.testing.assert_allclose(self.a, ref, rtol=self.tol, atol=self.tol)

    def test_geru(self):
        ref = self.alpha * cupy.outer(self.x, self.y) + self.a
        if self.mode is not None:
            self.alpha = self.mode.array(self.alpha)
        cublas.geru(self.alpha, self.x, self.y, self.a)
        cupy.testing.assert_allclose(self.a, ref, rtol=self.tol, atol=self.tol)

    def test_gerc(self):
        ref = self.alpha * cupy.outer(self.x, self.y.conj()) + self.a
        if self.mode is not None:
            self.alpha = self.mode.array(self.alpha)
        cublas.gerc(self.alpha, self.x, self.y, self.a)
        cupy.testing.assert_allclose(self.a, ref, rtol=self.tol, atol=self.tol)
