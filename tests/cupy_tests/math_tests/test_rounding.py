import unittest

from cupy import testing


@testing.gpu
class TestRounding(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_complex(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_raises(TypeError)
    def check_unary_complex_unsupported(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        getattr(xp, name)(a)

    @testing.for_dtypes(['?', 'b', 'h', 'i', 'q', 'e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_negative(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        return getattr(xp, name)(a)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_negative_complex(self, name, xp, dtype):
        a = xp.array([-3-3j, -2-2j, -1-1j, 1+1j, 2+2j, 3+3j], dtype=dtype)
        return getattr(xp, name)(a)

    def test_rint(self):
        self.check_unary('rint')
        self.check_unary_complex('rint')

    def test_rint_negative(self):
        self.check_unary_negative('rint')
        self.check_unary_negative_complex('rint')

    def test_floor(self):
        self.check_unary('floor')
        self.check_unary_complex_unsupported('floor')

    def test_ceil(self):
        self.check_unary('ceil')
        self.check_unary_complex_unsupported('ceil')

    def test_trunc(self):
        self.check_unary('trunc')
        self.check_unary_complex_unsupported('trunc')

    def test_fix(self):
        self.check_unary('fix')
        self.check_unary_complex_unsupported('fix')
