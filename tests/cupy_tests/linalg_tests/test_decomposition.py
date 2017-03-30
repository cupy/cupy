import unittest

import numpy

import cupy
from cupy import cuda
from cupy import testing


@unittest.skipUnless(
    cuda.cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
@testing.gpu
class TestCholeskyDecomposition(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_dtypes([
        numpy.int32, numpy.int64, numpy.uint32, numpy.uint64,
        numpy.float32, numpy.float64])
    @testing.numpy_cupy_allclose(atol=1e-3)
    def check_L(self, array, xp, dtype):
        a = xp.asarray(array, dtype=dtype)
        return xp.linalg.cholesky(a)

    def test_decomposition(self):
        # A normal positive definite matrix
        A = numpy.random.randint(0, 100, size=(5, 5))
        A = numpy.dot(A, A.transpose())
        self.check_L(A)
        # np.linalg.cholesky only uses a lower triangle of an array
        self.check_L(numpy.array([[1, 2], [1, 9]]))


@unittest.skipUnless(
    cuda.cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
@testing.gpu
class TestQRDecomposition(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_float_dtypes(no_float16=True)
    def check_mode(self, array, mode, dtype):
        a_cpu = numpy.asarray(array, dtype=dtype)
        a_gpu = cupy.asarray(array, dtype=dtype)
        result_cpu = numpy.linalg.qr(a_cpu, mode=mode)
        result_gpu = cupy.linalg.qr(a_gpu, mode=mode)
        if isinstance(result_cpu, tuple):
            for b_cpu, b_gpu in zip(result_cpu, result_gpu):
                cupy.testing.assert_allclose(b_cpu, b_gpu, atol=1e-4)
        else:
            cupy.testing.assert_allclose(result_cpu, result_gpu, atol=1e-4)

    def test_r_mode(self):
        self.check_mode(numpy.random.randn(2, 3), mode='r')
        self.check_mode(numpy.random.randn(3, 3), mode='r')
        self.check_mode(numpy.random.randn(4, 2), mode='r')

    def test_raw_mode(self):
        self.check_mode(numpy.random.randn(2, 4), mode='raw')
        self.check_mode(numpy.random.randn(2, 3), mode='raw')
        self.check_mode(numpy.random.randn(4, 5), mode='raw')

    def test_complete_mode(self):
        self.check_mode(numpy.random.randn(2, 4), mode='complete')
        self.check_mode(numpy.random.randn(2, 3), mode='complete')
        self.check_mode(numpy.random.randn(4, 5), mode='complete')

    def test_reduced_mode(self):
        self.check_mode(numpy.random.randn(2, 4), mode='reduced')
        self.check_mode(numpy.random.randn(2, 3), mode='reduced')
        self.check_mode(numpy.random.randn(4, 5), mode='reduced')
