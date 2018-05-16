import unittest

import cupy
from cupy import testing
import cupyx.scipy.special

try:
    import scipy.special
except ImportError:
    pass


@testing.gpu
@testing.with_requires('scipy')
class TestTrigonometric(unittest.TestCase):

    def _get_xp_func(self, xp):
        if xp is cupy:
            return cupyx.scipy.special
        else:
            return scipy.special

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(self._get_xp_func(xp), name)(a)

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

    def test_i1(self):
        self.check_unary('i1')
