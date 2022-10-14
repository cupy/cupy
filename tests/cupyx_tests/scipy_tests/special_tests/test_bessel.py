import unittest

import cupy
from cupy import testing
import cupyx.scipy.special  # NOQA


@testing.gpu
@testing.with_requires('scipy')
class TestSpecial:

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(rtol=1e-5, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
        import scipy.special  # NOQA

        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(scp.special, name)(a)

    def test_j0(self):
        self.check_unary('j0')

    def test_j1(self):
        self.check_unary('j1')

    def test_y0(self):
        self.check_unary('y0')

    def test_y1(self):
        self.check_unary('y1')

    def test_i0(self):
        self.check_unary('i0')

    def test_i0e(self):
        self.check_unary('i0e')

    def test_i1(self):
        self.check_unary('i1')

    def test_i1e(self):
        self.check_unary('i1e')

    def test_k0(self):
        self.check_unary('k0')

    def test_k0e(self):
        self.check_unary('k0e')

    def test_k1(self):
        self.check_unary('k1')

    def test_k1e(self):
        self.check_unary('k1e')

    @testing.for_dtypes('iId', name='order_dtype')
    @testing.for_dtypes('efd')
    @testing.numpy_cupy_allclose(atol=1e-12, scipy_name='scp')
    def test_yn(self, xp, scp, dtype, order_dtype):
        import scipy.special  # NOQA

        n = xp.arange(0, 10, dtype=order_dtype)
        a = xp.linspace(-10, 10, 100, dtype=dtype)
        return scp.special.yn(n[:, xp.newaxis], a[xp.newaxis, :])


@testing.gpu
@testing.with_requires('scipy')
class TestFusionSpecial(unittest.TestCase):

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(rtol=1e-5, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
        import scipy.special  # NOQA

        a = testing.shaped_arange((2, 3), xp, dtype)

        @cupy.fuse()
        def f(x):
            return getattr(scp.special, name)(x)

        return f(a)

    def test_j0(self):
        self.check_unary('j0')

    def test_j1(self):
        self.check_unary('j1')

    def test_y0(self):
        self.check_unary('y0')

    def test_y1(self):
        self.check_unary('y1')

    def test_i0(self):
        self.check_unary('i0')

    def test_i0e(self):
        self.check_unary('i0e')

    def test_i1(self):
        self.check_unary('i1')

    def test_i1e(self):
        self.check_unary('i1e')

    def test_k0(self):
        self.check_unary('k0')

    def test_k0e(self):
        self.check_unary('k0e')

    def test_k1(self):
        self.check_unary('k1')

    def test_k1e(self):
        self.check_unary('k1e')

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(rtol=1e-5, scipy_name='scp')
    def test_chbevl_dependent_fusion(self, dtype, xp, scp):
        @cupy.fuse
        def fused(x):
            _k0 = scp.special.k0(x)
            _k0e = scp.special.k0e(x)
            _k1 = scp.special.k1(x)
            _k1e = scp.special.k1e(x)

            # rgamma depends on `chbevl` too
            # see rgamma definition for additional info
            _rgamma = scp.special.rgamma(x)

            return _k0 + _k0e + _k1 + _k1e + _rgamma

        a = xp.linspace(-10, 10, 50, dtype=dtype).reshape((2, -1))
        return fused(a)
