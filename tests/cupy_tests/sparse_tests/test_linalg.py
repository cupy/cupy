import unittest

import numpy
try:
    import scipy.sparse
    import scipy.sparse.linalg  # NOQA
    import scipy.stats
    scipy_available = True
except ImportError:
    scipy_available = False

import cupy.sparse.linalg  # NOQA
from cupy import testing
from cupy.testing import condition


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
}))
@unittest.skipUnless(scipy_available, 'requires scipy')
class TestLsqr(unittest.TestCase):

    def setUp(self):
        rvs = scipy.stats.randint(0, 15).rvs
        self.A = scipy.sparse.random(100, 100, density=0.2, data_rvs=rvs).A
        self.b = scipy.sparse.random(1, 100, density=0.2, data_rvs=rvs).A
        self.b = numpy.squeeze(self.b)

    @testing.numpy_cupy_raises(sp_name='sp')
    def test_size(self, xp, sp):
        b = numpy.append(self.b, [1])
        A = xp.array(self.A).astype(self.dtype)
        b = xp.array(b).astype(self.dtype)
        sp.linalg.lsqr(A, b)

    @condition.retry(5)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_csrmatrix(self, xp, sp):
        A = xp.array(self.A).astype(self.dtype)
        b = xp.array(self.b).astype(self.dtype)
        A = sp.csr_matrix(A)
        x = sp.linalg.lsqr(A, b)
        if xp == numpy:
            return x[0].astype(self.dtype)
        else:
            return x

    @condition.retry(5)
    @testing.numpy_cupy_allclose(atol=1e-1, sp_name='sp')
    def test_ndarray(self, xp, sp):
        A = xp.array(self.A).astype(self.dtype)
        b = xp.array(self.b).astype(self.dtype)
        x = sp.linalg.lsqr(A, b)
        if xp == numpy:
            return x[0].astype(self.dtype)
        else:
            return x
