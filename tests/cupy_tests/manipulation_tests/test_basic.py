import unittest

from cupy import testing


@testing.gpu
class TestBasic(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype)
        b = xpy.empty((2, 3, 4), dtype=dtype)
        xpy.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_dtype(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype='?')
        b = xpy.empty((2, 3, 4), dtype=dtype)
        xpy.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_broadcast(self, xpy, dtype):
        a = testing.shaped_arange((3, 1), xpy, dtype)
        b = xpy.empty((2, 3, 4), dtype=dtype)
        xpy.copyto(b, a)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copyto_where(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype)
        b = testing.shaped_reverse_arange((2, 3, 4), xpy, dtype)
        c = testing.shaped_arange((2, 3, 4), xpy, '?')
        xpy.copyto(a, b, where=c)
        return a
