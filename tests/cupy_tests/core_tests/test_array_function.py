import unittest

import numpy
import six

import cupy
from cupy import testing


@testing.gpu
class TestArrayFunction(unittest.TestCase):

    def test_array_function(self):
        a = numpy.random.randn(100, 100)
        a_cpu = numpy.asarray(a)
        a_gpu = cupy.asarray(a)

        # The numpy call for both CPU and GPU arrays is intentional to test the
        # __array_function__ protocol
        qr_cpu = numpy.linalg.qr(a_cpu)
        qr_gpu = numpy.linalg.qr(a_gpu)

        if isinstance(qr_cpu, tuple):
            for b_cpu, b_gpu in six.moves.zip(qr_cpu, qr_gpu):
                self.assertEqual(b_cpu.dtype, b_gpu.dtype)
                cupy.testing.assert_allclose(b_cpu, b_gpu, atol=1e-4)
        else:
            self.assertEqual(qr_cpu.dtype, qr_gpu.dtype)
            cupy.testing.assert_allclose(qr_cpu, qr_gpu, atol=1e-4)
