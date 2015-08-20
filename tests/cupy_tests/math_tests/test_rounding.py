import unittest

from cupy import testing


@testing.gpu
class TestRounding(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_dtypes(['?', 'b', 'h', 'i', 'q', 'e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_negative(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        return getattr(xp, name)(a)

    def test_rint(self):
        self.check_unary('rint')

    def test_rint_negative(self):
        self.check_unary_negative('rint')

    def test_floor(self):
        self.check_unary('floor')

    def test_ceil(self):
        self.check_unary('ceil')

    def test_trunc(self):
        self.check_unary('trunc')
