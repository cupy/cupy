import unittest

import cupy
from cupy import cuda
from cupy import testing


@testing.gpu
class TestFromData(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array(self, xp, dtype):
        return xp.array([[1, 2, 3], [2, 3, 4]], dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_copy(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.array(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_array_copy_is_copied(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.array(a)
        a.fill(0)
        return b

    @testing.for_all_dtypes(name='dtype1')
    @testing.for_all_dtypes(name='dtype2')
    @testing.numpy_cupy_array_equal()
    def test_array_copy_with_dtype(self, xp, dtype1, dtype2):
        a = testing.shaped_arange((2, 3, 4), xp, dtype1)
        return xp.array(a, dtype=dtype2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asarray(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.asarray(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_asarray_is_not_copied(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.asarray(a)
        a.fill(0)
        return b

    def test_ascontiguousarray_on_noncontiguous_array(self):
        a = testing.shaped_arange((2, 3, 4))
        b = a.transpose(2, 0, 1)
        c = cupy.ascontiguousarray(b)
        self.assertTrue(c.flags.c_contiguous)
        testing.assert_array_equal(b, c)

    def test_ascontiguousarray_on_contiguous_array(self):
        a = testing.shaped_arange((2, 3, 4))
        b = cupy.ascontiguousarray(a)
        self.assertIs(a, b)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_copy(self, xp, dtype, order):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        b = a.copy(order=order)
        a[1] = 1
        return b

    @testing.multi_gpu(2)
    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    def test_copy_multigpu(self, dtype, order):
        with cuda.Device(0):
            src = cupy.random.uniform(-1, 1, (2, 3)).astype(dtype)
        with cuda.Device(1):
            dst = src.copy(order)
        testing.assert_allclose(src, dst, rtol=0, atol=0)
