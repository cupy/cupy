import unittest

from cupy import testing


@testing.gpu
class TestSplit(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_array_equal()
    def test_array_spliti1(self, xpy):
        a = testing.shaped_arange((3, 11), xpy)
        split = xpy.array_split(a, 4, 1)
        return xpy.concatenate(split, 1)

    @testing.numpy_cupy_array_equal()
    def test_array_spliti2(self, xpy):
        a = testing.shaped_arange((3, 11), xpy)
        split = xpy.array_split(a, 4, 1)
        return xpy.concatenate(split, -1)

    @testing.numpy_cupy_array_equal()
    def test_dsplit(self, xpy):
        a = testing.shaped_arange((3, 3, 12), xpy)
        split = xpy.dsplit(a, 4)
        return xpy.dstack(split)

    @testing.numpy_cupy_array_equal()
    def test_hsplit_vectors(self, xpy):
        a = testing.shaped_arange((12,), xpy)
        split = xpy.hsplit(a, 4)
        return xpy.hstack(split)

    @testing.numpy_cupy_array_equal()
    def test_hsplit(self, xpy):
        a = testing.shaped_arange((3, 12), xpy)
        split = xpy.hsplit(a, 4)
        return xpy.hstack(split)

    @testing.numpy_cupy_array_equal()
    def test_split_by_sections1(self, xpy):
        a = testing.shaped_arange((3, 11), xpy)
        split = xpy.split(a, (2, 4, 9), 1)
        return xpy.concatenate(split, 1)

    @testing.numpy_cupy_array_equal()
    def test_split_by_sections2(self, xpy):
        a = testing.shaped_arange((3, 11), xpy)
        split = xpy.split(a, (2, 4, 9), 1)
        return xpy.concatenate(split, -1)

    @testing.numpy_cupy_array_equal()
    def test_vsplit(self, xpy):
        a = testing.shaped_arange((12, 3), xpy)
        split = xpy.vsplit(a, 4)
        return xpy.vstack(split)
