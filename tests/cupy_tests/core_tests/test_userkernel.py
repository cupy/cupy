import unittest

import numpy

import cupy
from cupy import testing


class TestUserkernel(unittest.TestCase):

    def test_manual_indexing(self, n=100):
        in1 = cupy.random.uniform(-1, 1, n).astype(cupy.float32)
        in2 = cupy.random.uniform(-1, 1, n).astype(cupy.float32)
        uesr_kernel_1 = cupy.ElementwiseKernel(
            'T x, T y',
            'T z',
            '''
                z = x + y;
            ''',
            'uesr_kernel_1')
        out1 = uesr_kernel_1(in1, in2)

        uesr_kernel_2 = cupy.ElementwiseKernel(
            'raw T x, raw T y',
            'raw T z',
            '''
                z[i] = x[i] + y[i];
            ''',
            'uesr_kernel_2')
        out2 = uesr_kernel_2(in1, in2, size=n)

        testing.assert_array_equal(out1, out2)

    def test_python_scalar(self):
        for typ in (int, float, bool):
            dtype = numpy.dtype(typ).type
            in1_cpu = numpy.random.randint(0, 1, (4, 5)).astype(dtype)
            in1 = cupy.array(in1_cpu)
            scalar_value = typ(2)
            uesr_kernel_1 = cupy.ElementwiseKernel(
                'T x, T y',
                'T z',
                '''
                    z = x + y;
                ''',
                'uesr_kernel_1')
            out1 = uesr_kernel_1(in1, scalar_value)

            expected = in1_cpu + dtype(2)
            testing.assert_array_equal(out1, expected)

    @testing.for_all_dtypes()
    def test_numpy_scalar(self, dtype):
        in1_cpu = numpy.random.randint(0, 1, (4, 5)).astype(dtype)
        in1 = cupy.array(in1_cpu)
        scalar_value = dtype(2)
        uesr_kernel_1 = cupy.ElementwiseKernel(
            'T x, T y',
            'T z',
            '''
                z = x + y;
            ''',
            'uesr_kernel_1')
        out1 = uesr_kernel_1(in1, scalar_value)

        expected = in1_cpu + dtype(2)
        testing.assert_array_equal(out1, expected)


@testing.parameterize(*testing.product({
    'value': [-1, 2 ** 32, 2 ** 63 - 1, -(2 ** 63)],
}))
class TestUserkernelScalar(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_scalar(self, xp, dtype):
        x = testing.shaped_arange((2, 3, 4), xp, dtype)
        if xp is numpy:
            y = numpy.array(self.value).astype(dtype)
            return x + y
        else:
            kernel = cupy.ElementwiseKernel('T x, T y', 'T z', 'z = x + y')
            return kernel(x, self.value)
