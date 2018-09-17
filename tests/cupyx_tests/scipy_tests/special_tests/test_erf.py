import unittest

import cupy
from cupy import testing
import cupyx.scipy.special  # NOQA


class _TestBase(object):

    def test_erf(self):
        self.check_unary('erf')

    def test_erfc(self):
        self.check_unary('erfc')

    def test_erfcx(self):
        self.check_unary('erfcx')

    def test_erfinv(self):
        self.check_unary('erfinv')
        self.check_unary_random('erfinv', scale=2, offset=-1)
        self.check_unary_boundary('erfinv', boundary=-1)
        self.check_unary_boundary('erfinv', boundary=1)

    def test_erfcinv(self):
        self.check_unary('erfcinv')
        self.check_unary_random('erfcinv', scale=2, offset=0)
        self.check_unary_boundary('erfcinv', boundary=0)
        self.check_unary_boundary('erfcinv', boundary=2)


@testing.gpu
@testing.with_requires('scipy')
class TestSpecial(unittest.TestCase, _TestBase):

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
        import scipy.special  # NOQA

        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(scp.special, name)(a)

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def check_unary_random(self, name, xp, scp, dtype, scale, offset):
        import scipy.special  # NOQA

        a = testing.shaped_random((2, 3), xp, dtype, scale=scale) + offset
        return getattr(scp.special, name)(a)

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def check_unary_boundary(self, name, xp, scp, dtype, boundary):
        import scipy.special  # NOQA

        x = boundary * (1 - 1.0 / 1024)
        y = boundary
        z = boundary * (1 + 1.0 / 1024)
        a = xp.array([x, y, z], dtype=dtype)
        return getattr(scp.special, name)(a)


@testing.gpu
@testing.with_requires('scipy')
class TestFusionSpecial(unittest.TestCase, _TestBase):

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
        import scipy.special  # NOQA

        a = testing.shaped_arange((2, 3), xp, dtype)

        @cupy.fuse()
        def f(x):
            return getattr(scp.special, name)(x)

        return f(a)

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def check_unary_random(self, name, xp, scp, dtype, scale, offset):
        import scipy.special  # NOQA

        a = testing.shaped_random((2, 3), xp, dtype, scale=scale) + offset

        @cupy.fuse()
        def f(x):
            return getattr(scp.special, name)(x)

        return f(a)

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def check_unary_boundary(self, name, xp, scp, dtype, boundary):
        import scipy.special  # NOQA

        x = boundary * (1 - 1.0 / 1024)
        y = boundary
        z = boundary * (1 + 1.0 / 1024)
        a = xp.array([x, y, z], dtype=dtype)

        @cupy.fuse()
        def f(x):
            return getattr(scp.special, name)(x)

        return f(a)
