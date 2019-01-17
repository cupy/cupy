import unittest

from cupy import testing


@testing.gpu
class TestTrigonometric(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_binary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a, b)

    @testing.for_dtypes(['e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_unit(self, name, xp, dtype):
        a = xp.array([0.2, 0.4, 0.6, 0.8], dtype=dtype)
        return getattr(xp, name)(a)

    def test_sin(self):
        self.check_unary('sin')

    def test_cos(self):
        self.check_unary('cos')

    def test_tan(self):
        self.check_unary('tan')

    def test_arcsin(self):
        self.check_unary_unit('arcsin')

    def test_arccos(self):
        self.check_unary_unit('arccos')

    def test_arctan(self):
        self.check_unary('arctan')

    def test_arctan2(self):
        self.check_binary('arctan2')

    def test_hypot(self):
        self.check_binary('hypot')

    def test_deg2rad(self):
        self.check_unary('deg2rad')

    def test_rad2deg(self):
        self.check_unary('rad2deg')


@testing.gpu
class TestUnwrap(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_unwrap_1dim(self, xp, dtype):
        a = testing.shaped_random((5,), xp, dtype)
        return xp.unwrap(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_unwrap_1dim_with_discont(self, xp, dtype):
        a = testing.shaped_random((5,), xp, dtype)
        return xp.unwrap(a, discont=1.0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_unwrap_2dim_without_axis(self, xp, dtype):
        a = testing.shaped_random((4, 5), xp, dtype)
        return xp.unwrap(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_unwrap_2dim_with_axis(self, xp, dtype):
        a = testing.shaped_random((4, 5), xp, dtype)
        return xp.unwrap(a, axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_unwrap_2dim_with_discont(self, xp, dtype):
        a = testing.shaped_random((4, 5), xp, dtype)
        return xp.unwrap(a, discont=5.0, axis=1)
