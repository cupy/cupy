import unittest

import cupy

from cupy import testing


@testing.gpu
class TestJoin(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_array_equal()
    def test_column_stack(self, xpy, dtype1, dtype2):
        a = testing.shaped_arange((4, 3), xpy, dtype1)
        b = testing.shaped_arange((4,), xpy, dtype2)
        c = testing.shaped_arange((4, 2), xpy, dtype1)
        return xpy.column_stack((a, b, c))

    def test_column_stack_wrong_ndim1(self):
        a = cupy.zeros(())
        b = cupy.zeros((3,))
        with self.assertRaises(ValueError):
            cupy.column_stack((a, b))

    def test_column_stack_wrong_ndim2(self):
        a = cupy.zeros((3, 2, 3))
        b = cupy.zeros((3, 2))
        with self.assertRaises(ValueError):
            cupy.column_stack((a, b))

    def test_column_stack_wrong_shape(self):
        a = cupy.zeros((3, 2))
        b = cupy.zeros((4, 3))
        with self.assertRaises(ValueError):
            cupy.column_stack((a, b))

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate1(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype)
        b = testing.shaped_reverse_arange((2, 3, 2), xpy, dtype)
        c = testing.shaped_arange((2, 3, 3), xpy, dtype)
        return xpy.concatenate((a, b, c), axis=2)

    @testing.for_all_dtypes(name='dtype')
    @testing.numpy_cupy_array_equal()
    def test_concatenate2(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype)
        b = testing.shaped_reverse_arange((2, 3, 2), xpy, dtype)
        c = testing.shaped_arange((2, 3, 3), xpy, dtype)
        return xpy.concatenate((a, b, c), axis=-1)

    def test_concatenate_wrong_ndim(self):
        a = cupy.empty((2, 3))
        b = cupy.empty((2,))
        with self.assertRaises(ValueError):
            cupy.concatenate((a, b))

    def test_concatenate_wrong_shape(self):
        a = cupy.empty((2, 3, 4))
        b = cupy.empty((3, 3, 4))
        c = cupy.empty((4, 4, 4))
        with self.assertRaises(ValueError):
            cupy.concatenate((a, b, c))

    @testing.numpy_cupy_array_equal()
    def test_dstack(self, xpy):
        a = testing.shaped_arange((1, 3, 2), xpy)
        b = testing.shaped_arange((3,), xpy)
        c = testing.shaped_arange((1, 3), xpy)
        return xpy.dstack((a, b, c))

    @testing.numpy_cupy_array_equal()
    def test_hstack_vectors(self, xpy):
        a = xpy.arange(3)
        b = xpy.arange(2, -1, -1)
        return xpy.hstack((a, b))

    @testing.numpy_cupy_array_equal()
    def test_hstack(self, xpy):
        a = testing.shaped_arange((2, 1), xpy)
        b = testing.shaped_arange((2, 2), xpy)
        c = testing.shaped_arange((2, 3), xpy)
        return xpy.hstack((a, b, c))

    @testing.numpy_cupy_array_equal()
    def test_vstack_vectors(self, xpy):
        a = xpy.arange(3)
        b = xpy.arange(2, -1, -1)
        return xpy.vstack((a, b))

    def test_vstack_wrong_ndim(self):
        a = cupy.empty((3,))
        b = cupy.empty((3, 1))
        with self.assertRaises(ValueError):
            cupy.vstack((a, b))
