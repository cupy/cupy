import unittest

import numpy

import cupy
from cupy import cuda
from cupy import testing
from cupy.testing import condition


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

    @condition.retry(10)
    def test_solve(self):
        self.check_x((4, 4), (4,))
        self.check_x((5, 5), (5, 2))

    def test_invalid_shape(self):
        self.check_shape((2, 3), (4,))
        self.check_shape((3, 3), (2,))
        self.check_shape((3, 3, 4), (3,))


@testing.parameterize(*testing.product({
    'a_shape': [(2, 3, 6), (3, 4, 4, 3)],
    'dtype': [numpy.float32, numpy.float64],
    'axes': [None, (0, 2)],
}))
@testing.fix_random()
@testing.gpu
@unittest.skipUnless(
    cuda.cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
class TestTensorSolve(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        self.a = numpy.random.randint(
            0, 10, size=self.a_shape).astype(self.dtype)
        self.b = numpy.random.randint(
            0, 10, size=self.a_shape[:2]).astype(self.dtype)

    @testing.numpy_cupy_allclose(atol=0.02)
    def test_tensorsolve(self, xp):
        a = xp.array(self.a)
        b = xp.array(self.b)
        return xp.linalg.tensorsolve(a, b, axes=self.axes)


@unittest.skipUnless(
    cuda.cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
@testing.gpu
class TestInv(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_float_dtypes(no_float16=True)
    def check_x(self, a_shape, dtype):
        a_cpu = numpy.random.randint(0, 10, size=a_shape).astype(dtype)
        a_gpu = cupy.asarray(a_cpu)
        result_cpu = numpy.linalg.inv(a_cpu)
        result_gpu = cupy.linalg.inv(a_gpu)
        self.assertEqual(result_cpu.dtype, result_gpu.dtype)
        cupy.testing.assert_allclose(result_cpu, result_gpu, atol=1e-3)

    def check_shape(self, a_shape):
        a = cupy.random.rand(*a_shape)
        with self.assertRaises(numpy.linalg.LinAlgError):
            cupy.linalg.inv(a)

    @condition.retry(10)
    def test_inv(self):
        self.check_x((3, 3))
        self.check_x((4, 4))
        self.check_x((5, 5))

    def test_invalid_shape(self):
        self.check_shape((2, 3))
        self.check_shape((4, 1))


@unittest.skipUnless(
    cuda.cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
@testing.gpu
class TestPinv(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_float_dtypes(no_float16=True)
    def check_x(self, a_shape, rcond, dtype):
        a_cpu = numpy.random.randint(0, 10, size=a_shape).astype(dtype)
        a_gpu = cupy.asarray(a_cpu)
        result_cpu = numpy.linalg.pinv(a_cpu, rcond=rcond)
        result_gpu = cupy.linalg.pinv(a_gpu, rcond=rcond)

        self.assertEqual(result_cpu.dtype, result_gpu.dtype)
        cupy.testing.assert_allclose(result_cpu, result_gpu, atol=1e-3)

    def check_shape(self, a_shape, rcond):
        a = cupy.random.rand(*a_shape)
        with self.assertRaises(numpy.linalg.LinAlgError):
            cupy.linalg.pinv(a)

    @condition.retry(10)
    def test_pinv(self):
        self.check_x((3, 3), rcond=1e-15)
        self.check_x((2, 4), rcond=1e-15)
        self.check_x((3, 2), rcond=1e-15)

        self.check_x((4, 4), rcond=0.3)
        self.check_x((2, 5), rcond=0.5)
        self.check_x((5, 3), rcond=0.6)

    def test_invalid_shape(self):
        self.check_shape((2, 3, 4), rcond=1e-15)
        self.check_shape((2, 3, 4), rcond=0.5)
        self.check_shape((4, 3, 2, 1), rcond=1e-14)
        self.check_shape((4, 3, 2, 1), rcond=0.1)
