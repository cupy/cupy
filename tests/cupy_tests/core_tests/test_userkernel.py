import unittest

import cupy
from cupy import testing


@testing.gpu
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
