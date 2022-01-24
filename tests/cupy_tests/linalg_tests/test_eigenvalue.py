import numpy
import pytest

import cupy
from cupy.cuda import driver
from cupy.cuda import runtime
from cupy import testing


def _get_hermitian(xp, a, UPLO):
    if UPLO == 'U':
        return xp.triu(a) + xp.triu(a, 1).swapaxes(-2, -1).conj()
    else:
        return xp.tril(a) + xp.tril(a, -1).swapaxes(-2, -1).conj()


@testing.parameterize(*testing.product({
    'UPLO': ['U', 'L'],
}))
@pytest.mark.skipif(
    runtime.is_hip and driver.get_build_version() < 402,
    reason='eigensolver not added until ROCm 4.2.0')
class TestEigenvalue:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, contiguous_check=False)
    def test_eigh(self, xp, dtype):
        if xp == numpy and dtype == numpy.float16:
            # NumPy's eigh does not support float16
            _dtype = 'f'
        else:
            _dtype = dtype
        if numpy.dtype(_dtype).kind == 'c':
            a = xp.array([[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]], _dtype)
        else:
            a = xp.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], _dtype)
        w, v = xp.linalg.eigh(a, UPLO=self.UPLO)

        # Changed the verification method to check if Av and vw match, since
        # the eigenvectors of eigh() with CUDA 11.6 are mathematically correct
        # but may not match NumPy.
        A = _get_hermitian(xp, a, self.UPLO)
        if _dtype == numpy.float16:
            tol = 1e-3
        else:
            tol = 1e-5
        testing.assert_allclose(A @ v, v @ xp.diag(w), atol=tol, rtol=tol)
        # Check if v @ vt is an identity matrix
        testing.assert_allclose(v @ v.swapaxes(-2, -1).conj(),
                                xp.identity(A.shape[-1], _dtype), atol=tol,
                                rtol=tol)
        if xp == numpy and dtype == numpy.float16:
            w = w.astype('e')
        return w

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
        A = _get_hermitian(xp, a, self.UPLO)
        for i in range(a.shape[0]):
            testing.assert_allclose(
                A[i].dot(v[i]), w[i]*v[i], rtol=1e-5, atol=1e-5)
        return w

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
        A = _get_hermitian(xp, a, self.UPLO)
        for i in range(a.shape[0]):
            testing.assert_allclose(
                A[i].dot(v[i]), w[i]*v[i], rtol=1e-5, atol=1e-5)
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


@pytest.mark.parametrize('UPLO', ['U', 'L'])
@pytest.mark.parametrize('shape', [
    (0, 0),
    (2, 0, 0),
    (0, 3, 3),
])
@pytest.mark.skipif(
    runtime.is_hip and driver.get_build_version() < 402,
    reason='eigensolver not added until ROCm 4.2.0')
class TestEigenvalueEmpty:

    @testing.for_dtypes('ifdFD')
    @testing.numpy_cupy_allclose()
    def test_eigh(self, xp, dtype, shape, UPLO):
        a = xp.empty(shape, dtype)
        assert a.size == 0
        return xp.linalg.eigh(a, UPLO=UPLO)

    @testing.for_dtypes('ifdFD')
    @testing.numpy_cupy_allclose()
    def test_eigvalsh(self, xp, dtype, shape, UPLO):
        a = xp.empty(shape, dtype)
        assert a.size == 0
        return xp.linalg.eigvalsh(a, UPLO=UPLO)


@pytest.mark.parametrize('UPLO', ['U', 'L'])
@pytest.mark.parametrize('shape', [
    (),
    (3,),
    (2, 3),
    (4, 0),
    (2, 2, 3),
    (0, 2, 3),
])
@pytest.mark.skipif(
    runtime.is_hip and driver.get_build_version() < 402,
    reason='eigensolver not added until ROCm 4.2.0')
class TestEigenvalueInvalid:

    def test_eigh_shape_error(self, UPLO, shape):
        for xp in (numpy, cupy):
            a = xp.zeros(shape)
            with pytest.raises(numpy.linalg.LinAlgError):
                xp.linalg.eigh(a, UPLO)

    def test_eigvalsh_shape_error(self, UPLO, shape):
        for xp in (numpy, cupy):
            a = xp.zeros(shape)
            with pytest.raises(numpy.linalg.LinAlgError):
                xp.linalg.eigvalsh(a, UPLO)
