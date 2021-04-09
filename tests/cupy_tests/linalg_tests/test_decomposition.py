import unittest

import numpy
import pytest

import cupy
from cupy._core.internal import prod
from cupy import cusolver
from cupy import testing
from cupy.testing import _condition
import cupyx


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
    if dtype.kind == 'c':
        a = a + 1j * numpy.random.standard_normal(shape)
    u, s, vh = numpy.linalg.svd(a)
    if sym:
        assert m == n
        vh = u.conj().swapaxes(-1, -2)
    new_s = numpy.random.uniform(low_s, high_s, s.shape)
    new_a = numpy.einsum('...ij,...j,...jk->...ik', u, new_s, vh)
    if bias is not None:
        new_a += bias
    if dtype.kind in 'iu':
        new_a = numpy.rint(new_a)
    return new_a.astype(dtype)


@testing.gpu
class TestCholeskyDecomposition(unittest.TestCase):

    @testing.numpy_cupy_allclose(atol=1e-3)
    def check_L(self, array, xp):
        a = xp.asarray(array)
        return xp.linalg.cholesky(a)

    @testing.for_dtypes([
        numpy.int32, numpy.int64, numpy.uint32, numpy.uint64,
        numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
    def test_decomposition(self, dtype):
        # A positive definite matrix
        A = random_matrix((5, 5), dtype, scale=(10, 10000), sym=True)
        self.check_L(A)
        # np.linalg.cholesky only uses a lower triangle of an array
        self.check_L(numpy.array([[1, 2], [1, 9]], dtype))

    @testing.for_dtypes([
        numpy.int32, numpy.int64, numpy.uint32, numpy.uint64,
        numpy.float32, numpy.float64, numpy.complex64, numpy.complex128])
    def test_batched_decomposition(self, dtype):
        if not cusolver.check_availability('potrfBatched'):
            pytest.skip('potrfBatched is not available')
        Ab1 = random_matrix((3, 5, 5), dtype, scale=(10, 10000), sym=True)
        self.check_L(Ab1)
        Ab2 = random_matrix((2, 2, 5, 5), dtype, scale=(10, 10000), sym=True)
        self.check_L(Ab2)


@testing.gpu
class TestCholeskyInvalid(unittest.TestCase):

    def check_L(self, array):
        for xp in (numpy, cupy):
            a = xp.asarray(array)
            with cupyx.errstate(linalg='raise'):
                with pytest.raises(numpy.linalg.LinAlgError):
                    xp.linalg.cholesky(a)

    @testing.for_dtypes([
        numpy.int32, numpy.int64, numpy.uint32, numpy.uint64,
        numpy.float32, numpy.float64])
    def test_decomposition(self, dtype):
        A = numpy.array([[1, -2], [-2, 1]]).astype(dtype)
        self.check_L(A)


@testing.parameterize(*testing.product({
    'mode': ['r', 'raw', 'complete', 'reduced'],
}))
@testing.gpu
class TestQRDecomposition(unittest.TestCase):

    @testing.for_dtypes('fdFD')
    def check_mode(self, array, mode, dtype):
        a_cpu = numpy.asarray(array, dtype=dtype)
        a_gpu = cupy.asarray(array, dtype=dtype)
        result_cpu = numpy.linalg.qr(a_cpu, mode=mode)
        result_gpu = cupy.linalg.qr(a_gpu, mode=mode)
        if isinstance(result_cpu, tuple):
            for b_cpu, b_gpu in zip(result_cpu, result_gpu):
                assert b_cpu.dtype == b_gpu.dtype
                cupy.testing.assert_allclose(b_cpu, b_gpu, atol=1e-4)
        else:
            assert result_cpu.dtype == result_gpu.dtype
            cupy.testing.assert_allclose(result_cpu, result_gpu, atol=1e-4)

    @testing.fix_random()
    @_condition.repeat(3, 10)
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
@testing.gpu
class TestSVD(unittest.TestCase):

    def setUp(self):
        self.seed = testing.generate_seed()

    @testing.for_dtypes([
        numpy.int32, numpy.int64, numpy.uint32, numpy.uint64,
        numpy.float32, numpy.float64, numpy.complex64, numpy.complex128,
    ])
    def check_usv(self, shape, dtype):
        array = testing.shaped_random(
            shape, numpy, dtype=dtype, seed=self.seed)
        a_cpu = numpy.asarray(array, dtype=dtype)
        a_gpu = cupy.asarray(array, dtype=dtype)
        result_cpu = numpy.linalg.svd(a_cpu, full_matrices=self.full_matrices)
        result_gpu = cupy.linalg.svd(a_gpu, full_matrices=self.full_matrices)
        # Check if the input matrix is not broken
        cupy.testing.assert_allclose(a_gpu, a_cpu)

        assert len(result_gpu) == 3
        for i in range(3):
            assert result_gpu[i].shape == result_cpu[i].shape
            assert result_gpu[i].dtype == result_cpu[i].dtype
        u_cpu, s_cpu, vh_cpu = result_cpu
        u_gpu, s_gpu, vh_gpu = result_gpu
        cupy.testing.assert_allclose(s_gpu, s_cpu, rtol=1e-5, atol=1e-4)

        # reconstruct the matrix
        k = s_cpu.shape[-1]
        if len(shape) == 2:
            if self.full_matrices:
                a_gpu_usv = cupy.dot(u_gpu[:, :k] * s_gpu, vh_gpu[:k, :])
            else:
                a_gpu_usv = cupy.dot(u_gpu * s_gpu, vh_gpu)
        else:
            if self.full_matrices:
                a_gpu_usv = cupy.matmul(u_gpu[..., :k] * s_gpu[..., None, :],
                                        vh_gpu[..., :k, :])
            else:
                a_gpu_usv = cupy.matmul(u_gpu*s_gpu[..., None, :], vh_gpu)
        cupy.testing.assert_allclose(a_gpu, a_gpu_usv, rtol=1e-4, atol=1e-4)

        # assert unitary
        if len(shape) == 2:
            cupy.testing.assert_allclose(
                cupy.matmul(u_gpu.T.conj(), u_gpu),
                numpy.eye(u_gpu.shape[1]),
                atol=1e-4)
            cupy.testing.assert_allclose(
                cupy.matmul(vh_gpu, vh_gpu.T.conj()),
                numpy.eye(vh_gpu.shape[0]),
                atol=1e-4)
        else:
            batch = prod(shape[:-2])
            u_len = u_gpu.shape[-1]
            vh_len = vh_gpu.shape[-2]

            if batch == 0:
                id_u_cpu = numpy.empty(shape[:-2] + (u_len, u_len))
                id_vh_cpu = numpy.empty(shape[:-2] + (vh_len, vh_len))
            else:
                id_u_cpu = [numpy.eye(u_len) for _ in range(batch)]
                id_u_cpu = numpy.stack(id_u_cpu, axis=0).reshape(
                    *(shape[:-2]), u_len, u_len)
                id_vh_cpu = [numpy.eye(vh_len) for _ in range(batch)]
                id_vh_cpu = numpy.stack(id_vh_cpu, axis=0).reshape(
                    *(shape[:-2]), vh_len, vh_len)

            cupy.testing.assert_allclose(
                cupy.matmul(u_gpu.swapaxes(-1, -2).conj(), u_gpu),
                id_u_cpu, atol=1e-4)
            cupy.testing.assert_allclose(
                cupy.matmul(vh_gpu, vh_gpu.swapaxes(-1, -2).conj()),
                id_vh_cpu, atol=1e-4)

    @testing.for_dtypes([
        numpy.int32, numpy.int64, numpy.uint32, numpy.uint64,
        numpy.float32, numpy.float64, numpy.complex64, numpy.complex128,
    ])
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-4)
    def check_singular(self, shape, xp, dtype):
        array = testing.shaped_random(shape, xp, dtype=dtype, seed=self.seed)
        a = xp.asarray(array, dtype=dtype)
        a_copy = a.copy()
        result = xp.linalg.svd(
            a, full_matrices=self.full_matrices, compute_uv=False)
        # Check if the input matrix is not broken
        assert (a == a_copy).all()
        return result

    @_condition.repeat(3, 10)
    def test_svd_rank2(self):
        self.check_usv((3, 7))
        self.check_usv((2, 2))
        self.check_usv((7, 3))

    @_condition.repeat(3, 10)
    def test_svd_rank2_no_uv(self):
        self.check_singular((3, 7))
        self.check_singular((2, 2))
        self.check_singular((7, 3))

    @testing.with_requires('numpy>=1.16')
    def test_svd_rank2_empty_array(self):
        self.check_usv((0, 3))
        self.check_usv((3, 0))
        self.check_usv((1, 0))

    @testing.with_requires('numpy>=1.16')
    @testing.numpy_cupy_array_equal()
    def test_svd_rank2_empty_array_compute_uv_false(self, xp):
        array = xp.empty((3, 0))
        return xp.linalg.svd(
            array, full_matrices=self.full_matrices, compute_uv=False)

    @_condition.repeat(3, 10)
    def test_svd_rank3(self):
        self.check_usv((2, 3, 4))
        self.check_usv((2, 3, 7))
        self.check_usv((2, 4, 4))
        self.check_usv((2, 7, 3))
        self.check_usv((2, 4, 3))
        self.check_usv((2, 32, 32))  # still use _gesvdj_batched

    @_condition.repeat(3, 10)
    def test_svd_rank3_loop(self):
        # This tests the loop-based batched gesvd on CUDA (_gesvd_batched)
        self.check_usv((2, 64, 64))
        self.check_usv((2, 64, 32))
        self.check_usv((2, 32, 64))

    @_condition.repeat(3, 10)
    def test_svd_rank3_no_uv(self):
        self.check_singular((2, 3, 4))
        self.check_singular((2, 3, 7))
        self.check_singular((2, 4, 4))
        self.check_singular((2, 7, 3))
        self.check_singular((2, 4, 3))

    @_condition.repeat(3, 10)
    def test_svd_rank3_no_uv_loop(self):
        # This tests the loop-based batched gesvd on CUDA (_gesvd_batched)
        self.check_singular((2, 64, 64))
        self.check_singular((2, 64, 32))
        self.check_singular((2, 32, 64))

    @testing.with_requires('numpy>=1.16')
    def test_svd_rank3_empty_array(self):
        self.check_usv((0, 3, 4))
        self.check_usv((3, 0, 4))
        self.check_usv((3, 4, 0))
        self.check_usv((3, 0, 0))
        self.check_usv((0, 3, 0))
        self.check_usv((0, 0, 3))

    @testing.with_requires('numpy>=1.16')
    @testing.numpy_cupy_array_equal()
    def test_svd_rank3_empty_array_compute_uv_false1(self, xp):
        array = xp.empty((3, 0, 4))
        return xp.linalg.svd(
            array, full_matrices=self.full_matrices, compute_uv=False)

    @testing.with_requires('numpy>=1.16')
    @testing.numpy_cupy_array_equal()
    def test_svd_rank3_empty_array_compute_uv_false2(self, xp):
        array = xp.empty((0, 3, 4))
        return xp.linalg.svd(
            array, full_matrices=self.full_matrices, compute_uv=False)

    @_condition.repeat(3, 10)
    def test_svd_rank4(self):
        self.check_usv((2, 2, 3, 4))
        self.check_usv((2, 2, 3, 7))
        self.check_usv((2, 2, 4, 4))
        self.check_usv((2, 2, 7, 3))
        self.check_usv((2, 2, 4, 3))
        self.check_usv((2, 2, 32, 32))  # still use _gesvdj_batched

    @_condition.repeat(3, 10)
    def test_svd_rank4_loop(self):
        # This tests the loop-based batched gesvd on CUDA (_gesvd_batched)
        self.check_usv((3, 2, 64, 64))
        self.check_usv((3, 2, 64, 32))
        self.check_usv((3, 2, 32, 64))

    @_condition.repeat(3, 10)
    def test_svd_rank4_no_uv(self):
        self.check_singular((2, 2, 3, 4))
        self.check_singular((2, 2, 3, 7))
        self.check_singular((2, 2, 4, 4))
        self.check_singular((2, 2, 7, 3))
        self.check_singular((2, 2, 4, 3))

    @_condition.repeat(3, 10)
    def test_svd_rank4_no_uv_loop(self):
        # This tests the loop-based batched gesvd on CUDA (_gesvd_batched)
        self.check_singular((3, 2, 64, 64))
        self.check_singular((3, 2, 64, 32))
        self.check_singular((3, 2, 32, 64))

    @testing.with_requires('numpy>=1.16')
    def test_svd_rank4_empty_array(self):
        self.check_usv((0, 2, 3, 4))
        self.check_usv((1, 2, 0, 4))
        self.check_usv((1, 2, 3, 0))
