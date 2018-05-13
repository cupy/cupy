import sys
import unittest

import numpy
import six

import cupy
from cupy import core
from cupy import cuda
from cupy import testing


@testing.gpu
class TestElementwise(unittest.TestCase):

    def check_copy(self, dtype, src_id, dst_id):
        with cuda.Device(src_id):
            src = testing.shaped_arange((2, 3, 4), dtype=dtype)
        with cuda.Device(dst_id):
            dst = cupy.empty((2, 3, 4), dtype=dtype)
        core.elementwise_copy(src, dst)
        testing.assert_allclose(src, dst)

    @testing.for_all_dtypes()
    def test_copy(self, dtype):
        device_id = cuda.Device().id
        self.check_copy(dtype, device_id, device_id)

    @testing.multi_gpu(2)
    @testing.for_all_dtypes()
    def test_copy_multigpu(self, dtype):
        with self.assertRaises(ValueError):
            self.check_copy(dtype, 0, 1)

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_copy_zero_sized_array1(self, xp, dtype, order):
        src = xp.empty((0,), dtype=dtype)
        res = xp.copy(src, order=order)
        self.assertIsNot(src, res)
        return res

    @testing.for_orders('CFAK')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_copy_zero_sized_array2(self, xp, dtype, order):
        src = xp.empty((1, 0, 2), dtype=dtype)
        res = xp.copy(src, order=order)
        self.assertIsNot(src, res)
        return res

    @testing.for_orders('CFAK')
    def test_copy_orders(self, order):
        a = cupy.empty((2, 3, 4))
        b = cupy.copy(a, order)

        a_cpu = numpy.empty((2, 3, 4))
        b_cpu = numpy.copy(a_cpu, order)

        self.assertEqual(b.strides, b_cpu.strides)


@testing.gpu
class TestElementwiseInvalidArgument(unittest.TestCase):

    def test_invalid_kernel_name(self):
        with six.assertRaisesRegex(self, ValueError, 'Invalid kernel name'):
            cupy.ElementwiseKernel('T x', '', '', '1')


@testing.gpu
class TestElementwiseType(unittest.TestCase):

    # Skip this test due to NumPy bug on Windows (fixed in NumPy 1.14.0).
    # https://github.com/numpy/numpy/pull/9778
    @unittest.skipIf(
        sys.platform == 'win32' and testing.numpy_satisfies('<1.14'),
        'This test requires 1.14.0 when running on Windows.')
    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_upper_1(self, xp, dtype):
        a = xp.array([0], dtype=xp.int8)
        b = xp.iinfo(dtype).max
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_upper_2(self, xp, dtype):
        a = xp.array([1], dtype=xp.int8)
        b = xp.iinfo(dtype).max - 1
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_upper_3(self, xp, dtype):
        a = xp.array([xp.iinfo(dtype).max], dtype=dtype)
        b = xp.int8(0)
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_upper_4(self, xp, dtype):
        a = xp.array([xp.iinfo(dtype).max - 1], dtype=dtype)
        b = xp.int8(1)
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_lower_1(self, xp, dtype):
        a = xp.array([0], dtype=xp.int8)
        b = xp.iinfo(dtype).min
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_lower_2(self, xp, dtype):
        a = xp.array([-1], dtype=xp.int8)
        b = xp.iinfo(dtype).min + 1
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_lower_3(self, xp, dtype):
        a = xp.array([xp.iinfo(dtype).min], dtype=dtype)
        b = xp.int8(0)
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_lower_4(self, xp, dtype):
        a = xp.array([xp.iinfo(dtype).min + 1], dtype=dtype)
        b = xp.int8(-1)
        return a + b
