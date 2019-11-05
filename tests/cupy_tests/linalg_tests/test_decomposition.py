import unittest

import numpy
import six

import cupy
from cupy import cuda
from cupy import testing
from cupy.testing import condition


def random_matrix(shape, dtype, scale, sym=False):
    m, n = shape[-2:]
    dtype = numpy.dtype(dtype)
    assert dtype.kind in 'iufc'
    low_s, high_s = scale
    bias = None
    if dtype.kind in 'iu':
        # For an m \times n matrix M whose element is in [-0.5, 0.5], it holds
        # (singular value of M) <= \sqrt{mn} / 2
        err = numpy.sqrt(m * n) / 2.
        low_s += err
        high_s -= err
        if dtype.kind in 'u':
            assert sym, (
                'generating nonsymmetric matrix with uint cells is not'
                ' supported')
            # (singular value of numpy.ones((m, n))) <= \sqrt{mn}
            high_s = bias = high_s / (1 + numpy.sqrt(m * n))
    assert low_s <= high_s
    a = numpy.random.standard_normal(shape)
    u, s, vh = numpy.linalg.svd(a)
    new_s = numpy.random.uniform(low_s, high_s, s.shape)
    if sym:
        assert m == n
        new_a = numpy.einsum('...ij,...j,...kj', u, new_s, u)
    else:
        new_a = numpy.einsum('...ij,...j,...jk', u, new_s, vh)
    if bias is not None:
        new_a += bias
    if dtype.kind in 'iu':
        new_a = numpy.rint(new_a)
    return new_a.astype(dtype)


@unittest.skipUnless(
    cuda.cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
@testing.gpu
class TestCholeskyDecomposition(unittest.TestCase):

    @testing.numpy_cupy_allclose(atol=1e-3)
    def check_L(self, array, xp):
        a = xp.asarray(array)
        return xp.linalg.cholesky(a)

    @testing.for_dtypes([
        numpy.int32, numpy.int64, numpy.uint32, numpy.uint64,
        numpy.float32, numpy.float64])
    def test_decomposition(self, dtype):
        # A positive definite matrix
        A = random_matrix((5, 5), dtype, scale=(10, 10000), sym=True)
        self.check_L(A)
        # np.linalg.cholesky only uses a lower triangle of an array
        self.check_L(numpy.array([[1, 2], [1, 9]], dtype))


@testing.parameterize(*testing.product({
    'mode': ['r', 'raw', 'complete', 'reduced'],
}))
@unittest.skipUnless(
    cuda.cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
@testing.gpu
class TestQRDecomposition(unittest.TestCase):

    @testing.for_float_dtypes(no_float16=True)
    def check_mode(self, array, mode, dtype):
        a_cpu = numpy.asarray(array, dtype=dtype)
        a_gpu = cupy.asarray(array, dtype=dtype)
        result_cpu = numpy.linalg.qr(a_cpu, mode=mode)
        result_gpu = cupy.linalg.qr(a_gpu, mode=mode)
        if isinstance(result_cpu, tuple):
            for b_cpu, b_gpu in six.moves.zip(result_cpu, result_gpu):
                self.assertEqual(b_cpu.dtype, b_gpu.dtype)
                cupy.testing.assert_allclose(b_cpu, b_gpu, atol=1e-4)
        else:
            self.assertEqual(result_cpu.dtype, result_gpu.dtype)
            cupy.testing.assert_allclose(result_cpu, result_gpu, atol=1e-4)

    @testing.fix_random()
    @condition.repeat(3, 10)
    def test_mode(self):
        self.check_mode(numpy.random.randn(2, 4), mode=self.mode)
        self.check_mode(numpy.random.randn(3, 3), mode=self.mode)
        self.check_mode(numpy.random.randn(5, 4), mode=self.mode)

    @testing.with_requires('numpy>=1.16')
    def test_empty_array(self):
        self.check_mode(numpy.empty((0, 3)), mode=self.mode)
        self.check_mode(numpy.empty((3, 0)), mode=self.mode)


@testing.parameterize(*testing.product({
    'full_matrices': [True, False],
}))
@testing.fix_random()
@unittest.skipUnless(
    cuda.cusolver_enabled, 'Only cusolver in CUDA 8.0 is supported')
@testing.gpu
class TestSVD(unittest.TestCase):

    def setUp(self):
        self.seed = testing.generate_seed()

    @testing.for_float_dtypes(no_float16=True)
    def check_usv(self, shape, dtype):
        array = testing.shaped_random(
            shape, numpy, dtype=dtype, seed=self.seed)
        a_cpu = numpy.asarray(array, dtype=dtype)
        a_gpu = cupy.asarray(array, dtype=dtype)
        result_cpu = numpy.linalg.svd(a_cpu, full_matrices=self.full_matrices)
        result_gpu = cupy.linalg.svd(a_gpu, full_matrices=self.full_matrices)
        self.assertEqual(len(result_cpu), len(result_gpu))
        for b_cpu, b_gpu in zip(result_cpu, result_gpu):
            # Use abs to support an inverse vector
            cupy.testing.assert_allclose(
                numpy.abs(b_cpu), cupy.abs(b_gpu), atol=1e-4)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(atol=1e-4)
    def check_singular(self, shape, xp, dtype):
        array = testing.shaped_random(shape, xp, dtype=dtype, seed=self.seed)
        a = xp.asarray(array, dtype=dtype)
        a_copy = a.copy()
        result = xp.linalg.svd(
            a, full_matrices=self.full_matrices, compute_uv=False)
        # Check if the input matrix is not broken
        assert (a == a_copy).all()
        return result

    def check_rank2(self, array):
        with self.assertRaises(numpy.linalg.LinAlgError):
            cupy.linalg.svd(array, full_matrices=self.full_matrices)

    @condition.repeat(3, 10)
    def test_svd(self):
        self.check_usv((2, 3))
        self.check_usv((2, 2))
        self.check_usv((3, 2))

    @condition.repeat(3, 10)
    def test_svd_no_uv(self):
        self.check_singular((2, 3))
        self.check_singular((2, 2))
        self.check_singular((3, 2))

    @condition.repeat(3, 10)
    def test_rank2(self):
        self.check_rank2(cupy.random.randn(2, 3, 4).astype(numpy.float32))
        self.check_rank2(cupy.random.randn(1, 2, 3, 4).astype(numpy.float64))
