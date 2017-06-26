import unittest

import numpy

import cupy
from cupy import cuda
from cupy import testing


@unittest.skipUnless(
    cuda.cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
@testing.gpu
class TestSolve(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_float_dtypes(no_float16=True)
    def check_x(self, a_shape, b_shape, dtype):
        a_cpu = numpy.random.randint(0, 10, size=a_shape).astype(dtype)
        b_cpu = numpy.random.randint(0, 10, size=b_shape).astype(dtype)
        a_gpu = cupy.asarray(a_cpu)
        b_gpu = cupy.asarray(b_cpu)
        result_cpu = numpy.linalg.solve(a_cpu, b_cpu)
        result_gpu = cupy.linalg.solve(a_gpu, b_gpu)
        self.assertEqual(result_cpu.dtype, result_gpu.dtype)
        cupy.testing.assert_allclose(result_cpu, result_gpu, atol=1e-3)

    def check_shape(self, a_shape, b_shape):
        a = cupy.random.rand(*a_shape)
        b = cupy.random.rand(*b_shape)
        with self.assertRaises(numpy.linalg.LinAlgError):
            cupy.linalg.solve(a, b)

    def test_solve(self):
        self.check_x((4, 4), (4,))
        self.check_x((5, 5), (5, 2))

    def test_invalid_shape(self):
        self.check_shape((2, 3), (4,))
        self.check_shape((3, 3), (2,))
        self.check_shape((3, 3, 4), (3,))
