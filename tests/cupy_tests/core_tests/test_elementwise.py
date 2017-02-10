import unittest

import numpy

import cupy
from cupy import core
from cupy import cuda
from cupy import testing


@testing.gpu
class TestElementwise(unittest.TestCase):

    _multiprocess_can_split_ = True

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

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_copy_zero_sized_array1(self, xp, dtype, order):
        src = xp.empty((0,), dtype=dtype)
        return xp.copy(src, order=order)

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_copy_zero_sized_array2(self, xp, dtype, order):
        src = xp.empty((1, 0, 2), dtype=dtype)
        return xp.copy(src, order=order)

    @testing.for_CF_orders()
    def test_copy_orders(self, order):
        a = cupy.empty((2, 3, 4))
        b = cupy.copy(a, order)

        a_cpu = numpy.empty((2, 3, 4))
        b_cpu = numpy.copy(a_cpu, order)

        self.assertEqual(b.strides, b_cpu.strides)
