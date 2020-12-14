import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestArrayFunction(unittest.TestCase):

    @testing.with_requires('numpy>=1.17.0')
    def test_array_function(self):
        a = numpy.random.randn(100, 100)
        a_cpu = numpy.asarray(a)
        a_gpu = cupy.asarray(a)

        # The numpy call for both CPU and GPU arrays is intentional to test the
        # __array_function__ protocol
        qr_cpu = numpy.linalg.qr(a_cpu)
        qr_gpu = numpy.linalg.qr(a_gpu)

        if isinstance(qr_cpu, tuple):
            for b_cpu, b_gpu in zip(qr_cpu, qr_gpu):
                assert b_cpu.dtype == b_gpu.dtype
                cupy.testing.assert_allclose(b_cpu, b_gpu, atol=1e-4)
        else:
            assert qr_cpu.dtype == qr_gpu.dtype
            cupy.testing.assert_allclose(qr_cpu, qr_gpu, atol=1e-4)

    @testing.with_requires('numpy>=1.17.0')
    def test_array_function2(self):
        a = numpy.random.randn(100, 100)
        a_cpu = numpy.asarray(a)
        a_gpu = cupy.asarray(a)

        # The numpy call for both CPU and GPU arrays is intentional to test the
        # __array_function__ protocol
        out_cpu = numpy.sum(a_cpu, axis=1)
        out_gpu = numpy.sum(a_gpu, axis=1)

        assert out_cpu.dtype == out_gpu.dtype
        cupy.testing.assert_allclose(out_cpu, out_gpu, atol=1e-4)

    @testing.with_requires('numpy>=1.17.0')
    @testing.numpy_cupy_equal()
    def test_array_function_can_cast(self, xp):
        return numpy.can_cast(xp.arange(2), 'f4')

    @testing.with_requires('numpy>=1.17.0')
    @testing.numpy_cupy_equal()
    def test_array_function_common_type(self, xp):
        return numpy.common_type(xp.arange(2, dtype='f8'),
                                 xp.arange(2, dtype='f4'))

    @testing.with_requires('numpy>=1.17.0')
    @testing.numpy_cupy_equal()
    def test_array_function_result_type(self, xp):
        return numpy.result_type(3, xp.arange(2, dtype='f8'))
