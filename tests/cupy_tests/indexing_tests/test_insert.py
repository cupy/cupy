import unittest

from cupy import testing


@testing.gpu
class TestInsert(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_fill_diagonal_shape1(self, xp, dtype):
        a = testing.shaped_arange((3, 3), xp, dtype)
        xp.fill_diag(a, val=0)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_fill_diagonal_shape2(self, xp, dtype):
        a = testing.shaped_arange((2, 2, 2), xp, dtype)
        xp.fill_diag(a, val=1)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_fill_diagonal_wrap1(self, xp, dtype):
        a = testing.shaped_arange((3, 5), xp, dtype)
        xp.fill_diag(a, val=0, wrap=True)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_fill_diagonal_wrap2(self, xp, dtype):
        a = testing.shaped_arange((5, 3), xp, dtype)
        xp.fill_diag(a, val=0, wrap=True)
        return a
