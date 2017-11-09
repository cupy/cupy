import unittest

import numpy
from scipy.sparse import linalg

from cupy import testing
from cupy.sparse import linalg


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@testing.with_requires('scipy')
class TestLsqr(unittest.TestCase):

    def setUp(self):
        self.A = numpy.random.randint(2, size=(10, 10))
        self.b = numpy.random.randint(2, size=10)

    @testing.numpy_cupy_allclose(atol=1e-3, sp_name='sp')
    def test_csrmatrix(self, xp, sp):
        A = xp.array(self.A).astype(self.dtype)
        b = xp.array(self.b).astype(self.dtype)
        A = A + A.T
        A = sp.csr_matrix(A)
        x = sp.linalg.lsqr(A, b)
        if xp == numpy:
            x = x[0]
            x = x.astype(self.dtype)
        return x

    @testing.numpy_cupy_allclose(atol=1e-3,  sp_name='sp')
    def test_ndarray(self, xp, sp):
        A = xp.array(self.A).astype(self.dtype)
        b = xp.array(self.b).astype(self.dtype)
        A = A + A.T
        x = sp.linalg.lsqr(A, b)
        if xp == numpy:
            x = x[0]
            x = x.astype(self.dtype)
        return x
