import unittest

import cupy
from cupy import testing
import cupyx.scipy.special

import numpy

try:
    import scipy.special
    _scipy_available = True
except ImportError:
    _scipy_available = False


@testing.gpu
@testing.with_requires('scipy')
class TestSpecial(unittest.TestCase):

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
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

    def test_i1(self):
        self.check_unary('i1')


@testing.gpu
@testing.with_requires('scipy')
class TestFusionSpecial(unittest.TestCase):

    @testing.for_dtypes(['f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5, scipy_name='scp')
    def check_unary(self, name, xp, scp, dtype):
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

    def test_i1(self):
        self.check_unary('i1')


class TestFusionCPUSpecial(unittest.TestCase):

    @testing.for_dtypes(['f', 'd'])
    def check_unary(self, name, dtype):
        a = testing.shaped_arange((2, 3), numpy, dtype)

        @cupy.fuse()
        def f(x):
            return getattr(cupyx.scipy.special, name)(x)

        if _scipy_available:
            x = getattr(scipy.special, name)(a)
            numpy.testing.assert_array_equal(f(a), x)
        else:
            with self.assertRaises(ImportError):
                f(a)

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
