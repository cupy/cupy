import unittest

import cupy
from cupy import testing


@testing.gpu
class TestDims(unittest.TestCase):

    _multiprocess_can_split_ = True

    def check_atleast(self, func, xpy):
        a = testing.shaped_arange((), xpy)
        b = testing.shaped_arange((2,), xpy)
        c = testing.shaped_arange((2, 2), xpy)
        d = testing.shaped_arange((4, 3, 2), xpy)
        return func(a, b, c, d)

    @testing.numpy_cupy_array_list_equal()
    def test_atleast_1d1(self, xpy):
        return self.check_atleast(xpy.atleast_1d, xpy)

    @testing.numpy_cupy_array_equal()
    def test_atleast_1d2(self, xpy):
        a = testing.shaped_arange((1, 3, 2), xpy)
        return xpy.atleast_1d(a)

    @testing.numpy_cupy_array_list_equal()
    def test_atleast_2d1(self, xpy):
        return self.check_atleast(xpy.atleast_2d, xpy)

    @testing.numpy_cupy_array_equal()
    def test_atleast_2d2(self, xpy):
        a = testing.shaped_arange((1, 3, 2), xpy)
        return xpy.atleast_2d(a)

    @testing.numpy_cupy_array_list_equal()
    def test_atleast_3d1(self, xpy):
        return self.check_atleast(xpy.atleast_3d, xpy)

    @testing.numpy_cupy_array_equal()
    def test_atleast_3d2(self, xpy):
        a = testing.shaped_arange((1, 3, 2), xpy)
        return xpy.atleast_3d(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_broadcast_arrays(self, xpy, dtype):
        a = testing.shaped_arange((2, 1, 3, 4), xpy, dtype)
        b = testing.shaped_arange((3, 1, 4), xpy, dtype)
        c, d = xpy.broadcast_arrays(a, b)
        return d

    def test_broadcast(self):
        a = testing.shaped_arange((2, 1, 3, 4))
        b = testing.shaped_arange((3, 1, 4))
        bc = cupy.broadcast(a, b)
        self.assertEqual((2, 3, 3, 4), bc.shape)
        self.assertEqual(2 * 3 * 3 * 4, bc.size)
        self.assertEqual(4, bc.nd)

    @testing.numpy_cupy_array_equal()
    def test_squeeze(self, xpy):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xpy)
        return a.squeeze()

    @testing.numpy_cupy_array_equal()
    def test_squeeze_along_axis(self, xpy):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xpy)
        return a.squeeze(axis=2)

    def test_squeeze_failure(self):
        a = testing.shaped_arange((2, 1, 3, 4))
        with self.assertRaises(RuntimeError):
            a.squeeze(axis=2)

    @testing.numpy_cupy_array_equal()
    def test_external_squeeze(self, xpy):
        a = testing.shaped_arange((1, 2, 1, 3, 1, 1, 4, 1), xpy)
        return xpy.squeeze(a)
