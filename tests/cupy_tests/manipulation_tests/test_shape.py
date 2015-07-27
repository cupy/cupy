import unittest

from cupy import testing


@testing.gpu
class TestShape(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_array_equal()
    def test_reshape_strides(self, xpy):
        a = testing.shaped_arange((1, 1, 1, 2, 2))
        return a.strides

    @testing.numpy_cupy_array_equal()
    def test_reshape2(self, xpy):
        a = xpy.zeros((8,), dtype=xpy.float32)
        return a.reshape((1, 1, 1, 4, 1, 2)).strides

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_nocopy_reshape(self, xpy, dtype):
        a = xpy.zeros((2, 3, 4), dtype=dtype)
        b = a.reshape(4, 3, 2)
        b[1] = 1
        return a

    @testing.numpy_cupy_array_equal()
    def test_transposed_reshape(self, xpy):
        a = testing.shaped_arange((2, 3, 4), xpy).T
        return a.reshape(4, 6)

    @testing.numpy_cupy_array_equal()
    def test_transposed_reshape2(self, xpy):
        a = testing.shaped_arange((2, 3, 4), xpy).transpose(2, 0, 1)
        return a.reshape(2, 3, 4)

    @testing.numpy_cupy_array_equal()
    def test_reshape_with_unknown_dimension(self, xpy):
        a = testing.shaped_arange((2, 3, 4), xpy)
        return a.reshape(3, -1)

    def test_reshape_with_multiple_unknown_dimensions(self):
        a = testing.shaped_arange((2, 3, 4))
        with self.assertRaises(ValueError):
            a.reshape(3, -1, -1)

    @testing.numpy_cupy_array_equal()
    def test_ravel(self, xpy):
        a = testing.shaped_arange((2, 3, 4), xpy)
        a = a.transpose(2, 0, 1)
        return a.ravel()
