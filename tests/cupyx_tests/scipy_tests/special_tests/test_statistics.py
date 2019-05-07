import unittest

import cupy
from cupy import testing
import cupyx.scipy.special  # NOQA


class _TestBase(object):

    def test_ndtr(self):
        self.check_unary('ndtr')


@testing.gpu
@testing.with_requires('scipy')
class TestSpecial(unittest.TestCase, _TestBase):

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
        import scipy.special  # NOQA

        a = testing.shaped_arange((2, 3), xp, dtype)
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
