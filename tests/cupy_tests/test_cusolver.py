import unittest

import numpy
import pytest

import cupy
from cupy import cusolver
from cupy import testing
from cupy.testing import attr


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
@attr.gpu
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
@attr.gpu
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
