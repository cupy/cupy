import unittest

import cupy
from cupy import testing


@testing.gpu
class TestIndexing(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_take_by_scalar(self, xp):
        a = testing.shaped_arange((2, 4, 3), xp)
        return a.take(2, axis=1)

    @testing.numpy_cupy_array_equal()
    def test_external_take_by_scalar(self, xp):
        a = testing.shaped_arange((2, 4, 3), xp)
        return xp.take(a, 2, axis=1)

    @testing.numpy_cupy_array_equal()
    def test_take_by_array(self, xp):
        a = testing.shaped_arange((2, 4, 3), xp)
        b = xp.array([[1, 3], [2, 0]])
        return a.take(b, axis=1)

    @testing.numpy_cupy_array_equal()
    def test_take_no_axis(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        b = xp.array([[10, 5], [3, 20]])
        return a.take(b)

    @testing.with_requires('numpy>=1.15')
    @testing.numpy_cupy_array_equal()
    def test_take_along_axis(self, xp):
        a = testing.shaped_random((2, 4, 3), xp, dtype='float32')
        b = testing.shaped_random((2, 6, 3), xp, dtype='int64', scale=4)
        return xp.take_along_axis(a, b, axis=-2)

    @testing.with_requires('numpy>=1.15')
    @testing.numpy_cupy_array_equal()
    def test_take_along_axis_none_axis(self, xp):
        a = testing.shaped_random((2, 4, 3), xp, dtype='float32')
        b = testing.shaped_random((30,), xp, dtype='int64', scale=24)
        return xp.take_along_axis(a, b, axis=None)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return a.diagonal(1, 2, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_external_diagonal(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return xp.diagonal(a, 1, 2, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal_negative1(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return a.diagonal(-1, 2, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal_negative2(self, xp, dtype):
        a = testing.shaped_arange((3, 3, 3), xp, dtype)
        return a.diagonal(0, -1, -2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal_negative3(self, xp, dtype):
        a = testing.shaped_arange((3, 3, 3), xp, dtype)
        return a.diagonal(0, -1, 1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal_negative4(self, xp, dtype):
        a = testing.shaped_arange((3, 3, 3), xp, dtype)
        return a.diagonal(0, -3, -1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal_negative5(self, xp, dtype):
        a = testing.shaped_arange((3, 3, 3), xp, dtype)
        return a.diagonal(0, -1, -3)

    @testing.with_requires('numpy>=1.15')
    @testing.numpy_cupy_raises()
    def test_diagonal_invalid1(self, xp):
        a = testing.shaped_arange((3, 3, 3), xp)
        a.diagonal(0, 1, 3)

    @testing.with_requires('numpy>=1.15')
    @testing.numpy_cupy_raises()
    def test_diagonal_invalid2(self, xp):
        a = testing.shaped_arange((3, 3, 3), xp)
        a.diagonal(0, 2, -4)


@testing.gpu
class TestChoose(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_choose(self, xp, dtype):
        a = xp.array([0, 2, 1, 2])
        c = testing.shaped_arange((3, 4), xp, dtype)
        return a.choose(c)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_choose_broadcast(self, xp, dtype):
        a = xp.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        c = xp.array([-10, 10], dtype=dtype)
        return a.choose(c)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_choose_broadcast2(self, xp, dtype):
        a = xp.array([0, 1])
        c = testing.shaped_arange((3, 5, 2), xp, dtype)
        return a.choose(c)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_choose_wrap(self, xp, dtype):
        a = xp.array([0, 3, -1, 5])
        c = testing.shaped_arange((3, 4), xp, dtype)
        return a.choose(c, mode='wrap')

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_choose_clip(self, xp, dtype):
        a = xp.array([0, 3, -1, 5])
        c = testing.shaped_arange((3, 4), xp, dtype)
        return a.choose(c, mode='clip')

    def test_unknown_clip(self):
        a = cupy.array([0, 3, -1, 5])
        c = testing.shaped_arange((3, 4), cupy, cupy.float32)
        with self.assertRaises(TypeError):
            a.choose(c, mode='unknow')

    def test_raise(self):
        a = cupy.array([2])
        c = cupy.array([[0, 1]])
        with self.assertRaises(ValueError):
            a.choose(c)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_choose_broadcast_fail(self, xp, dtype):
        a = xp.array([0, 1])
        c = testing.shaped_arange((3, 5, 4), xp, dtype)
        return a.choose(c)
