import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestBasic(unittest.TestCase):

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty(self, xp, dtype, order):
        a = xp.empty((2, 3, 4), dtype=dtype, order=order)
        a.fill(0)
        return a

    @testing.slow
    def test_empty_huge_size(self):
        a = cupy.empty((1024, 2048, 1024), dtype='b')
        a.fill(123)
        self.assertTrue((a == 123).all())
        # Free huge memory for slow test
        del a
        cupy.get_default_memory_pool().free_all_blocks()

    @testing.slow
    def test_empty_huge_size_fill0(self):
        a = cupy.empty((1024, 2048, 1024), dtype='b')
        a.fill(0)
        self.assertTrue((a == 0).all())
        # Free huge memory for slow test
        del a
        cupy.get_default_memory_pool().free_all_blocks()

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_scalar(self, xp, dtype, order):
        a = xp.empty(None, dtype=dtype, order=order)
        a.fill(0)
        return a

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_int(self, xp, dtype, order):
        a = xp.empty(3, dtype=dtype, order=order)
        a.fill(0)
        return a

    @testing.slow
    def test_empty_int_huge_size(self):
        a = cupy.empty(2 ** 31, dtype='b')
        a.fill(123)
        self.assertTrue((a == 123).all())
        # Free huge memory for slow test
        del a
        cupy.get_default_memory_pool().free_all_blocks()

    @testing.slow
    def test_empty_int_huge_size_fill0(self):
        a = cupy.empty(2 ** 31, dtype='b')
        a.fill(0)
        self.assertTrue((a == 0).all())
        # Free huge memory for slow test
        del a
        cupy.get_default_memory_pool().free_all_blocks()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a)
        b.fill(0)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_C_order(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        a = xp.asfortranarray(a)
        b = xp.empty_like(a, order='C')
        b.fill(0)
        self.assertTrue(b.flags.c_contiguous)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_lowercase(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        a = xp.asfortranarray(a)
        b = xp.empty_like(a, order='c')
        b.fill(0)
        self.assertTrue(b.flags.c_contiguous)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_F_order(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a, order='F')
        b.fill(0)
        self.assertTrue(b.flags.f_contiguous)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_A_order(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a, order='A')
        b.fill(0)
        self.assertTrue(b.flags.c_contiguous)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_A_order2(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        a = xp.asfortranarray(a)
        b = xp.empty_like(a, order='A')
        b.fill(0)
        self.assertTrue(b.flags.f_contiguous)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_A_order3(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a[:, ::2, :], order='A')
        b.fill(0)
        self.assertTrue(b.flags.c_contiguous)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_K_order(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a, order='K')
        b.fill(0)
        self.assertTrue(b.flags.c_contiguous)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_K_order2(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        a = xp.asfortranarray(a)
        b = xp.empty_like(a, order='K')
        b.fill(0)
        self.assertTrue(b.flags.f_contiguous)
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like_K_order3(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        # test strides that are both non-contiguous and non-descending
        a = a[:, ::2, :].swapaxes(0, 1)
        b = xp.empty_like(a, order='K')
        b.fill(0)
        self.assertFalse(b.flags.c_contiguous)
        self.assertFalse(b.flags.f_contiguous)
        return b

    @testing.for_all_dtypes()
    def test_empty_like_K_strides(self, dtype):
        # test strides that are both non-contiguous and non-descending
        a = testing.shaped_arange((2, 3, 4), numpy, dtype)
        a = a[:, ::2, :].swapaxes(0, 1)
        b = numpy.empty_like(a, order='K')
        b.fill(0)

        # GPU case
        ag = testing.shaped_arange((2, 3, 4), cupy, dtype)
        ag = ag[:, ::2, :].swapaxes(0, 1)
        bg = cupy.empty_like(ag, order='K')
        bg.fill(0)

        # make sure NumPy and CuPy strides agree
        self.assertEqual(b.strides, bg.strides)
        return

    @testing.numpy_cupy_raises()
    def test_empty_like_invalid_order(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a, order='Q')
        return b

    @testing.for_CF_orders()
    def test_empty_zero_sized_array_strides(self, order):
        a = numpy.empty((1, 0, 2), dtype='d', order=order)
        b = cupy.empty((1, 0, 2), dtype='d', order=order)
        self.assertEqual(b.strides, a.strides)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_eye(self, xp, dtype):
        return xp.eye(5, 4, 1, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_identity(self, xp, dtype):
        return xp.identity(4, dtype)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros(self, xp, dtype, order):
        return xp.zeros((2, 3, 4), dtype=dtype, order=order)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros_scalar(self, xp, dtype, order):
        return xp.zeros(None, dtype=dtype, order=order)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros_int(self, xp, dtype, order):
        return xp.zeros(3, dtype=dtype, order=order)

    @testing.for_CF_orders()
    def test_zeros_strides(self, order):
        a = numpy.zeros((2, 3), dtype='d', order=order)
        b = cupy.zeros((2, 3), dtype='d', order=order)
        self.assertEqual(b.strides, a.strides)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros_like(self, xp, dtype):
        a = xp.ndarray((2, 3, 4), dtype=dtype)
        return xp.zeros_like(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_ones(self, xp, dtype):
        return xp.ones((2, 3, 4), dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_ones_like(self, xp, dtype):
        a = xp.ndarray((2, 3, 4), dtype=dtype)
        return xp.ones_like(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_full(self, xp, dtype):
        return xp.full((2, 3, 4), 1, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    @testing.with_requires('numpy>=1.12.0')
    def test_full_default_dtype(self, xp, dtype):
        return xp.full((2, 3, 4), xp.array(1, dtype=dtype))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    @testing.with_requires('numpy>=1.12.0')
    def test_full_default_dtype_cpu_input(self, xp, dtype):
        return xp.full((2, 3, 4), numpy.array(1, dtype=dtype))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_full_like(self, xp, dtype):
        a = xp.ndarray((2, 3, 4), dtype=dtype)
        return xp.full_like(a, 1)
