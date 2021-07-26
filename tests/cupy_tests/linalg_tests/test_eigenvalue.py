import unittest

import numpy
import pytest

import cupy
from cupy.cuda import driver
from cupy.cuda import runtime
from cupy import testing


@testing.parameterize(*testing.product({
    'UPLO': ['U', 'L'],
}))
@testing.gpu
@pytest.mark.skipif(
    runtime.is_hip and driver.get_build_version() < 402,
    reason='eigensolver not added until ROCm 4.2.0')
class TestEigenvalue(unittest.TestCase):

    @testing.for_all_dtypes(no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, contiguous_check=False)
    def test_eigh(self, xp, dtype):
        a = xp.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], dtype)
        w, v = xp.linalg.eigh(a, UPLO=self.UPLO)
        # NumPy, cuSOLVER, rocSOLVER all sort in ascending order,
        # so they should be directly comparable
        return w, v

    @testing.for_all_dtypes(no_bool=True, no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, contiguous_check=False)
    def test_eigh_batched(self, xp, dtype):
        a = xp.array([[[1, 0, 3], [0, 5, 0], [7, 0, 9]],
                      [[3, 0, 3], [0, 7, 0], [7, 0, 11]]], dtype)
        w, v = xp.linalg.eigh(a, UPLO=self.UPLO)

        # NumPy, cuSOLVER, rocSOLVER all sort in ascending order,
        # so w's should be directly comparable. However, both cuSOLVER
        # and rocSOLVER pick a different convention for constructing
        # eigenvectors, so v's are not directly comparible and we verify
        # them through the eigen equation A*v=w*v.
        A = xp.triu(a) if self.UPLO == 'U' else xp.tril(a)
        A = A + A.swapaxes(-2, -1)
        for i in range(a.shape[0]):
            testing.assert_allclose(
                (A[i] - xp.diag(a[i].diagonal())).dot(v[i]),
                w[i]*v[i],
                rtol=1e-5, atol=1e-5)
        return w

    def test_eigh_float16(self):
        # NumPy's eigh deos not support float16
        a = cupy.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], 'e')
        w, v = cupy.linalg.eigh(a, UPLO=self.UPLO)

        assert w.dtype == numpy.float16
        assert v.dtype == numpy.float16

        na = numpy.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], 'f')
        nw, nv = numpy.linalg.eigh(na, UPLO=self.UPLO)

        testing.assert_allclose(w, nw, rtol=1e-3, atol=1e-4)
        testing.assert_allclose(v, nv, rtol=1e-3, atol=1e-4)

    @testing.for_dtypes('FD')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, contiguous_check=False)
    def test_eigh_complex(self, xp, dtype):
        a = xp.array([[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]], dtype)
        w, v = xp.linalg.eigh(a, UPLO=self.UPLO)

        # NumPy, cuSOLVER, rocSOLVER all sort in ascending order,
        # so w's should be directly comparable. However,
        # rocSOLVER seems to pick a different convention in eigenvectors,
        # so v's are not directly comparible
        if runtime.is_hip:
            A = xp.triu(a) if self.UPLO == 'U' else xp.tril(a)
            A = A + A.swapaxes(-2, -1).conj() - xp.diag(a.diagonal())
            testing.assert_allclose(
                A.dot(v), w*v, rtol=1e-5, atol=1e-5)
            return w
        else:
            return w, v

    @testing.for_dtypes('FD')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, contiguous_check=False)
    def test_eigh_complex_batched(self, xp, dtype):
        a = xp.array([[[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]],
                      [[0, 2j, 3], [4j, 4, 6j], [7, 8j, 8]]], dtype)
        w, v = xp.linalg.eigh(a, UPLO=self.UPLO)

        # NumPy, cuSOLVER, rocSOLVER all sort in ascending order,
        # so w's should be directly comparable. However, both cuSOLVER
        # and rocSOLVER pick a different convention for constructing
        # eigenvectors, so v's are not directly comparible and we verify
        # them through the eigen equation A*v=w*v.
        A = xp.triu(a) if self.UPLO == 'U' else xp.tril(a)
        A = A + A.swapaxes(-2, -1).conj()
        for i in range(a.shape[0]):
            testing.assert_allclose(
                (A[i] - xp.diag(a[i].diagonal())).dot(v[i]),
                w[i]*v[i],
                rtol=1e-5, atol=1e-5)
        return w

    @testing.for_all_dtypes(no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_eigvalsh(self, xp, dtype):
        a = xp.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], dtype)
        w = xp.linalg.eigvalsh(a, UPLO=self.UPLO)
        # NumPy, cuSOLVER, rocSOLVER all sort in ascending order,
        # so they should be directly comparable
        return w

    @testing.for_all_dtypes(no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_eigvalsh_batched(self, xp, dtype):
        a = xp.array([[[1, 0, 3], [0, 5, 0], [7, 0, 9]],
                      [[3, 0, 3], [0, 7, 0], [7, 0, 11]]], dtype)
        w = xp.linalg.eigvalsh(a, UPLO=self.UPLO)
        # NumPy, cuSOLVER, rocSOLVER all sort in ascending order,
        # so they should be directly comparable
        return w

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_eigvalsh_complex(self, xp, dtype):
        a = xp.array([[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]], dtype)
        w = xp.linalg.eigvalsh(a, UPLO=self.UPLO)
        # NumPy, cuSOLVER, rocSOLVER all sort in ascending order,
        # so they should be directly comparable
        return w

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_eigvalsh_complex_batched(self, xp, dtype):
        a = xp.array([[[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]],
                      [[0, 2j, 3], [4j, 4, 6j], [7, 8j, 8]]], dtype)
        w = xp.linalg.eigvalsh(a, UPLO=self.UPLO)
        # NumPy, cuSOLVER, rocSOLVER all sort in ascending order,
        # so they should be directly comparable
        return w
