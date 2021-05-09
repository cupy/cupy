import unittest

import numpy
import pytest

import cupy
from cupy import cusolver
from cupy import testing
from cupy.testing import _attr
from cupy._core import _routines_linalg as _linalg
import cupyx


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'shape': [
        # gesvdj tests
        (5, 3), (4, 4), (3, 5),
        # gesvdjBatched tests
        (2, 5, 3), (2, 4, 4), (2, 3, 5),
    ],
    'order': ['C', 'F'],
    'full_matrices': [True, False],
    'overwrite_a': [True, False],
}))
@_attr.gpu
class TestGesvdj(unittest.TestCase):

    def setUp(self):
        if not cusolver.check_availability('gesvdj'):
            pytest.skip('gesvdj is not available')
        shape = self.shape
        if self.dtype == numpy.complex64:
            a_real = numpy.random.random(shape).astype(numpy.float32)
            a_imag = numpy.random.random(shape).astype(numpy.float32)
            self.a = a_real + 1.j * a_imag
        elif self.dtype == numpy.complex128:
            a_real = numpy.random.random(shape).astype(numpy.float64)
            a_imag = numpy.random.random(shape).astype(numpy.float64)
            self.a = a_real + 1.j * a_imag
        else:
            self.a = numpy.random.random(shape).astype(self.dtype)

    def test_gesvdj(self):
        a = cupy.array(self.a, order=self.order)
        u, s, v = cusolver.gesvdj(a, full_matrices=self.full_matrices,
                                  overwrite_a=self.overwrite_a)

        # sigma = diag(s)
        shape = self.shape
        mn = min(shape[-2:])
        if self.full_matrices:
            sigma_shape = shape
        else:
            sigma_shape = shape[:-2] + (mn, mn)
        sigma = cupy.zeros(sigma_shape, self.dtype)
        ix = numpy.arange(mn)
        sigma[..., ix, ix] = s

        vh = v.swapaxes(-2, -1).conjugate()
        aa = cupy.matmul(cupy.matmul(u, sigma), vh)
        if self.dtype in (numpy.float32, numpy.complex64):
            decimal = 5
        else:
            decimal = 10
        testing.assert_array_almost_equal(aa, self.a, decimal=decimal)

    def test_gesvdj_no_uv(self):
        a = cupy.array(self.a, order=self.order)
        s = cusolver.gesvdj(a, full_matrices=self.full_matrices,
                            compute_uv=False, overwrite_a=self.overwrite_a)
        expect = numpy.linalg.svd(self.a, full_matrices=self.full_matrices,
                                  compute_uv=False)
        if self.dtype in (numpy.float32, numpy.complex64):
            decimal = 5
        else:
            decimal = 10
        testing.assert_array_almost_equal(s, expect, decimal=decimal)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'shape': [(5, 4), (1, 4, 3), (4, 3, 2)],
}))
@_attr.gpu
class TestGesvda(unittest.TestCase):

    def setUp(self):
        if not cusolver.check_availability('gesvda'):
            pytest.skip('gesvda is not available')
        if self.dtype == numpy.complex64:
            a_real = numpy.random.random(self.shape).astype(numpy.float32)
            a_imag = numpy.random.random(self.shape).astype(numpy.float32)
            self.a = a_real + 1.j * a_imag
        elif self.dtype == numpy.complex128:
            a_real = numpy.random.random(self.shape).astype(numpy.float64)
            a_imag = numpy.random.random(self.shape).astype(numpy.float64)
            self.a = a_real + 1.j * a_imag
        else:
            self.a = numpy.random.random(self.shape).astype(self.dtype)

    def test_gesvda(self):
        a = cupy.array(self.a)
        u, s, v = cusolver.gesvda(a)
        if a.ndim == 2:
            batch_size = 1
            a = a.reshape((1,) + a.shape)
            u = u.reshape((1,) + u.shape)
            s = s.reshape((1,) + s.shape)
            v = v.reshape((1,) + v.shape)
        else:
            batch_size = a.shape[0]
        for i in range(batch_size):
            sigma = cupy.diag(s[i])
            vh = v[i].T.conjugate()
            aa = cupy.matmul(cupy.matmul(u[i], sigma), vh)
            if self.dtype in (numpy.float32, numpy.complex64):
                decimal = 5
            else:
                decimal = 10
            testing.assert_array_almost_equal(aa, a[i], decimal=decimal)

    def test_gesvda_no_uv(self):
        a = cupy.array(self.a)
        s = cusolver.gesvda(a, compute_uv=False)
        expect = numpy.linalg.svd(self.a, compute_uv=False)
        if self.dtype in (numpy.float32, numpy.complex64):
            decimal = 5
        else:
            decimal = 10
        testing.assert_array_almost_equal(s, expect, decimal=decimal)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'order': ['C', 'F'],
    'UPLO': ['L', 'U'],
}))
@_attr.gpu
class TestSyevj(unittest.TestCase):

    def setUp(self):
        if not cusolver.check_availability('syevj'):
            pytest.skip('syevj is not available')
        if self.dtype in (numpy.complex64, numpy.complex128):
            self.a = numpy.array(
                [[1, 2j, 3], [-2j, 5, 4j], [3, -4j, 9]], dtype=self.dtype)
        else:
            self.a = numpy.array(
                [[1, 2, 3], [2, 5, 4], [3, 4, 9]], dtype=self.dtype)

    def test_syevj(self):
        a = cupy.array(self.a, order=self.order)
        w, v = cusolver.syevj(a, UPLO=self.UPLO, with_eigen_vector=True)

        # check eignvalue equation
        testing.assert_allclose(self.a.dot(
            v.get()), w * v, rtol=1e-3, atol=1e-4)

    def test_syevjBatched(self):
        lda, m = self.a.shape

        na = numpy.stack([self.a, self.a + numpy.diag(numpy.ones(m))])
        a = cupy.array(na, order=self.order)

        w, v = cusolver.syevj(a, UPLO=self.UPLO, with_eigen_vector=True)

        # check eignvalue equation
        batch_size = a.shape[0]
        for i in range(batch_size):
            testing.assert_allclose(
                na[i].dot(v[i].get()), w[i] * v[i], rtol=1e-3, atol=1e-4)

        # check arbitrary batch dimension shape
        na = numpy.stack([na, na, na])
        a = cupy.array(na, order=self.order)

        w, v = cusolver.syevj(a, UPLO=self.UPLO, with_eigen_vector=True)

        assert v.shape == a.shape
        assert w.shape == a.shape[:-1]


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'n': [10, 100],
    'nrhs': [None, 1, 10],
    'compute_type': [None,
                     _linalg.COMPUTE_TYPE_FP16,
                     _linalg.COMPUTE_TYPE_TF32,
                     _linalg.COMPUTE_TYPE_FP32],
}))
@_attr.gpu
class TestGesv(unittest.TestCase):
    _tol = {'f': 1e-5, 'd': 1e-12}

    def _make_random_matrix(self, shape, xp):
        a = testing.shaped_random(shape, xp, dtype=self.r_dtype, scale=1)
        if self.dtype.char in 'FD':
            a = a + 1j * testing.shaped_random(
                shape, xp, dtype=self.r_dtype, scale=1)
        return a

    def _make_well_conditioned_matrix(self, shape):
        a = self._make_random_matrix(shape, numpy)
        u, s, vh = numpy.linalg.svd(a)
        s = testing.shaped_random(s.shape, numpy, dtype=self.r_dtype,
                                  scale=1) + 1
        a = numpy.einsum('...ik,...k,...kj->...ij', u, s, vh)
        return cupy.array(a)

    def setUp(self):
        if not cusolver.check_availability('gesv'):
            pytest.skip('gesv is not available')
        self.dtype = numpy.dtype(self.dtype)
        self.r_dtype = self.dtype.char.lower()
        a = self._make_well_conditioned_matrix((self.n, self.n))
        if self.nrhs is None:
            x_shape = (self.n, )
        else:
            x_shape = (self.n, self.nrhs)
        self.x_ref = self._make_random_matrix(x_shape, cupy)
        b = numpy.dot(a, self.x_ref)
        self.tol = self._tol[self.r_dtype]
        self.a = cupy.array(a)
        self.b = cupy.array(b)
        if self.compute_type is not None:
            self.old_compute_type = _linalg.get_compute_type(self.dtype)
            _linalg.set_compute_type(self.dtype, self.compute_type)

    def tearDown(self):
        if self.compute_type is not None:
            _linalg.set_compute_type(self.dtype, self.old_compute_type)

    def test_gesv(self):
        x = cusolver.gesv(self.a, self.b)
        cupy.testing.assert_allclose(x, self.x_ref,
                                     rtol=self.tol, atol=self.tol)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'shape': [(32, 32), (37, 32)],
    'nrhs': [None, 1, 4],
    'compute_type': [None,
                     _linalg.COMPUTE_TYPE_FP16,
                     _linalg.COMPUTE_TYPE_TF32,
                     _linalg.COMPUTE_TYPE_FP32],
}))
@_attr.gpu
class TestGels(unittest.TestCase):
    _tol = {'f': 1e-5, 'd': 1e-12}

    def setUp(self):
        if not cusolver.check_availability('gels'):
            pytest.skip('gels is not available')
        if self.compute_type is not None:
            self.old_compute_type = _linalg.get_compute_type(self.dtype)
            _linalg.set_compute_type(self.dtype, self.compute_type)

    def tearDown(self):
        if self.compute_type is not None:
            _linalg.set_compute_type(self.dtype, self.old_compute_type)

    def test_gels(self):
        b_shape = [self.shape[0]]
        if self.nrhs is not None:
            b_shape.append(self.nrhs)
        a = testing.shaped_random(self.shape, numpy, dtype=self.dtype)
        b = testing.shaped_random(b_shape, numpy, dtype=self.dtype)
        tol = self._tol[numpy.dtype(self.dtype).char.lower()]
        x_lstsq = numpy.linalg.lstsq(a, b, rcond=None)[0]
        x_gels = cusolver.gels(cupy.array(a), cupy.array(b))
        cupy.testing.assert_allclose(x_gels, x_lstsq, rtol=tol, atol=tol)


@testing.parameterize(*testing.product({
    'tol': [0, 1e-5],
    'reorder': [0, 1, 2, 3],
}))
@testing.with_requires('scipy')
class TestCsrlsvqr(unittest.TestCase):

    n = 8
    density = 0.75
    _test_tol = {'f': 1e-5, 'd': 1e-12}

    def setUp(self):
        if not cusolver.check_availability('csrlsvqr'):
            pytest.skip('csrlsvqr is not available')

    def _setup(self, dtype):
        dtype = numpy.dtype(dtype)
        a_shape = (self.n, self.n)
        a = testing.shaped_random(a_shape, numpy, dtype=dtype, scale=2/self.n)
        a_mask = testing.shaped_random(a_shape, numpy, dtype='f', scale=1)
        a[a_mask > self.density] = 0
        a_diag = numpy.diag(numpy.ones((self.n,), dtype=dtype))
        a = a + a_diag
        b = testing.shaped_random((self.n,), numpy, dtype=dtype)
        test_tol = self._test_tol[dtype.char.lower()]
        return a, b, test_tol

    @testing.for_dtypes('fdFD')
    def test_csrlsvqr(self, dtype):
        a, b, test_tol = self._setup(dtype)
        ref_x = numpy.linalg.solve(a, b)
        cp_a = cupy.array(a)
        sp_a = cupyx.scipy.sparse.csr_matrix(cp_a)
        cp_b = cupy.array(b)
        x = cupy.cusolver.csrlsvqr(sp_a, cp_b, tol=self.tol,
                                   reorder=self.reorder)
        cupy.testing.assert_allclose(x, ref_x, rtol=test_tol,
                                     atol=test_tol)
