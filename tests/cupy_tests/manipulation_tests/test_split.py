import unittest

from cupy import testing


@testing.gpu
class TestSplit(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_array_split1(self, xp):
        a = testing.shaped_arange((3, 11), xp)
        return xp.array_split(a, 4, 1)

    @testing.numpy_cupy_array_equal()
    def test_array_split2(self, xp):
        a = testing.shaped_arange((3, 11), xp)
        return xp.array_split(a, 4, -1)

    @testing.numpy_cupy_array_equal()
    def test_array_split_empty_array(self, xp):
        a = testing.shaped_arange((5, 0), xp)
        return xp.array_split(a, [2, 4], 0)

    @testing.numpy_cupy_array_equal()
    def test_array_split_empty_sections(self, xp):
        a = testing.shaped_arange((3, 11), xp)
        return xp.array_split(a, [])

    @testing.numpy_cupy_array_equal()
    def test_array_split_out_of_bound1(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.array_split(a, [3])

    @testing.numpy_cupy_array_equal()
    def test_array_split_out_of_bound2(self, xp):
        a = testing.shaped_arange((0,), xp)
        return xp.array_split(a, [1])

    @testing.numpy_cupy_array_equal()
    def test_array_split_unordered_sections(self, xp):
        a = testing.shaped_arange((5,), xp)
        return xp.array_split(a, [4, 2])

    @testing.numpy_cupy_array_equal()
    def test_array_split_non_divisible(self, xp):
        a = testing.shaped_arange((5, 3), xp)
        return xp.array_split(a, 4)

    @testing.numpy_cupy_array_equal()
    def test_dsplit(self, xp):
        a = testing.shaped_arange((3, 3, 12), xp)
        return xp.dsplit(a, 4)

    @testing.numpy_cupy_array_equal()
    def test_hsplit_vectors(self, xp):
        a = testing.shaped_arange((12,), xp)
        return xp.hsplit(a, 4)

    @testing.numpy_cupy_array_equal()
    def test_hsplit(self, xp):
        a = testing.shaped_arange((3, 12), xp)
        return xp.hsplit(a, 4)

    @testing.numpy_cupy_array_equal()
    def test_split_by_sections1(self, xp):
        a = testing.shaped_arange((3, 11), xp)
        return xp.split(a, (2, 4, 9), 1)

    @testing.numpy_cupy_array_equal()
    def test_split_by_sections2(self, xp):
        a = testing.shaped_arange((3, 11), xp)
        return xp.split(a, (2, 4, 9), -1)

    @testing.numpy_cupy_array_equal()
    def test_split_by_sections3(self, xp):
        a = testing.shaped_arange((3, 11), xp)
        return xp.split(a, (-9, 4, -2), 1)

    @testing.numpy_cupy_array_equal()
    def test_split_out_of_bound1(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        return xp.split(a, [3])

    @testing.numpy_cupy_array_equal()
    def test_split_out_of_bound2(self, xp):
        a = testing.shaped_arange((0,), xp)
        return xp.split(a, [1])

    @testing.numpy_cupy_array_equal()
    def test_split_unordered_sections(self, xp):
        a = testing.shaped_arange((5,), xp)
        return xp.split(a, [4, 2])

    @testing.numpy_cupy_array_equal()
    def test_vsplit(self, xp):
        a = testing.shaped_arange((12, 3), xp)
        return xp.vsplit(a, 4)
