import unittest

from cupy import testing


@testing.gpu
class TestHyperbolic(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_unit(self, name, xp, dtype):
        a = xp.array([0.2, 0.4, 0.6, 0.8], dtype=dtype)
        return getattr(xp, name)(a)

    def test_sinh(self):
        self.check_unary('sinh')

    def test_cosh(self):
        self.check_unary('cosh')

    def test_tanh(self):
        self.check_unary('tanh')

    def test_arcsinh(self):
        self.check_unary('arcsinh')

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_arccosh(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype=dtype)
        return xp.arccosh(a)

    def test_arctanh(self):
        self.check_unary_unit('arctanh')
