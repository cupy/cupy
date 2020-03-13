import unittest

import numpy

import cupy
from cupy import testing
from cupy import cusolver


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64, numpy.complex64, numpy.complex128],
    'shape': [(5, 3), (4, 4), (3, 5)],
    'order': ['C', 'F'],
    'full_matrices': [True, False],
    'overwrite_a': [True, False],
}))
class TestGesvdj(unittest.TestCase):

    def setUp(self):
        m, n = self.shape
        if self.dtype == numpy.complex64:
            a_real = numpy.random.random((m, n)).astype(numpy.float32)
            a_imag = numpy.random.random((m, n)).astype(numpy.float32)
            self.a = a_real + 1.j * a_imag
        elif self.dtype == numpy.complex128:
            a_real = numpy.random.random((m, n)).astype(numpy.float64)
            a_imag = numpy.random.random((m, n)).astype(numpy.float64)
            self.a = a_real + 1.j * a_imag
        else:
            self.a = numpy.random.random((m, n)).astype(self.dtype)

    def test_gesvdj(self):
        a = cupy.array(self.a, order=self.order)
        u, s, v = cusolver.gesvdj(a, full_matrices=self.full_matrices,
                                  overwrite_a=self.overwrite_a)
        m, n = self.shape
        mn = min(m, n)
        if self.full_matrices:
            sigma = numpy.zeros((m, n), dtype=self.dtype)
            for i in range(mn):
                sigma[i][i] = s[i]
            sigma = cupy.array(sigma)
        else:
            sigma = cupy.diag(s)
        if self.dtype in (numpy.complex64, numpy.complex128):
            vh = v.T.conjugate()
        else:
            vh = v.T
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

