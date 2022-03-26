import unittest

import cupy
import numpy
from cupy import testing
import cupyx.scipy.special  # NOQA
if cupyx.scipy._scipy_available:
    import scipy.special


# TODO: After the feature is released
# requires_scipy_special_backend = testing.with_requires('scipy>=1.x.x')
requires_scipy_special_backend = unittest.skip(
    'scipy.special backend feature has not been released'
)


@testing.gpu
@testing.with_requires('scipy')
class TestSpecial:

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
        import scipy.special  # NOQA

        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(scp.special, name)(a)

    def test_j0(self):
        self.check_unary('j0')
        self.check_scipy_backend('j0')

    def test_j1(self):
        self.check_unary('j1')
        self.check_scipy_backend('j1')

    def test_y0(self):
        self.check_unary('y0')
        self.check_scipy_backend('y0')

    def test_y1(self):
        self.check_unary('y1')
        self.check_scipy_backend('y1')

    def test_i0(self):
        self.check_unary('i0')
        self.check_scipy_backend('i0')

    def test_i0e(self):
        self.check_unary('i0e')

    def test_i1(self):
        self.check_unary('i1')
        self.check_scipy_backend('i1')

    def test_i1e(self):
        self.check_unary('i1e')

    @testing.for_dtypes('iId', name='order_dtype')
    @testing.for_dtypes('efd')
    @testing.numpy_cupy_allclose(atol=1e-12, scipy_name='scp')
    def test_yn(self, xp, scp, dtype, order_dtype):
        import scipy.special  # NOQA

        n = xp.arange(0, 10, dtype=order_dtype)
        a = xp.linspace(-10, 10, 100, dtype=dtype)
        return scp.special.yn(n[:, xp.newaxis], a[xp.newaxis, :])

    @requires_scipy_special_backend
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_scipy_backend(self, name, xp):
        a = testing.shaped_arange((2, 3), xp)
        backend = 'scipy' if xp is numpy else cupyx.scipy.special
        with scipy.special.set_backend(backend):
            out = getattr(scipy.special, name)(a)
        return out


@testing.gpu
@testing.with_requires('scipy')
class TestFusionSpecial(unittest.TestCase):

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
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
