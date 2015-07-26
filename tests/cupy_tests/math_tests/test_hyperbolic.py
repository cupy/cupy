import unittest

from cupy import testing


@testing.gpu
class TestHyperbolic(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary(self, name, xpy, dtype):
        a = testing.shaped_arange((2, 3), xpy, dtype)
        return getattr(xpy, name)(a)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_unit(self, name, xpy, dtype):
        a = xpy.array([0.2, 0.4, 0.6, 0.8], dtype=dtype)
        return getattr(xpy, name)(a)

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
    def test_arccosh(self, xpy, dtype):
        a = xpy.array([1, 2, 3], dtype=dtype)
        return xpy.arccosh(a)

    def test_arctanh(self):
        self.check_unary_unit('arctanh')
