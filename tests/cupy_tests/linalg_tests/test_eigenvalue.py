import numpy
import pytest

import cupy
from cupy.cuda import driver
from cupy.cuda import runtime
from cupy import testing
from cupyx import cusolver


def _get_hermitian(xp, a, UPLO):
    if UPLO == 'U':
        return xp.triu(a) + xp.triu(a, 1).swapaxes(-2, -1).conj()
    else:
        return xp.tril(a) + xp.tril(a, -1).swapaxes(-2, -1).conj()


def _real_to_complex(x):
    if x.dtype == 'float32':
        return x.astype(numpy.complex64)
    elif x.dtype == 'float64':
        return x.astype(numpy.complex128)
    else:
        assert numpy.iscomplexobj(x)
        return x


@testing.parameterize(*testing.product({
    'UPLO': ['U', 'L'],
}))
@pytest.mark.skipif(
    runtime.is_hip and driver.get_build_version() < 402,
    reason='eigensolver not added until ROCm 4.2.0')
class TestSymEigenvalue:

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
        # eigenvectors, so v's are not directly comparable and we verify
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
        # eigenvectors, so v's are not directly comparable and we verify
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


@pytest.mark.skipif(runtime.is_hip, reason="hip does not support eig")
@pytest.mark.skipif(
    cupy.cuda.runtime.runtimeGetVersion() < 12060,
    reason='Requires CUDA 12.6+')
class TestEigenvalue:
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, contiguous_check=False)
    def test_eig(self, xp, dtype):
        if not cusolver.check_availability('geev'):
            pytest.skip('geev is not available')
        if numpy.dtype(dtype).kind == 'c':
            a = xp.array([[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]], dtype)
        else:
            a = xp.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], dtype)
        w, v = xp.linalg.eig(a)
        tol = 1e-5
        testing.assert_allclose(a @ v, v @ xp.diag(w), atol=tol, rtol=tol)
        w = _real_to_complex(w)
        # Canonicalize the order
        return xp.sort(w)

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, contiguous_check=False)
    def test_eig_hermitian(self, xp, dtype):
        if not cusolver.check_availability('geev'):
            pytest.skip('geev is not available')
        if numpy.dtype(dtype).kind == 'c':
            a = xp.array([[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]], dtype)
        else:
            a = xp.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], dtype)
        a = _get_hermitian(xp, a, 'U')
        w, v = xp.linalg.eig(a)
        tol = 1e-5
        testing.assert_allclose(a @ v, v @ xp.diag(w), atol=tol, rtol=tol)
        w = _real_to_complex(w)
        # Canonicalize the order
        return xp.sort(w)

    @testing.for_all_dtypes(no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_eigvals(self, xp, dtype):
        if not cusolver.check_availability('geev'):
            pytest.skip('geev is not available')
        a = xp.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], dtype)
        w = xp.linalg.eigvals(a)
        w = _real_to_complex(w)
        # Canonicalize the order
        return xp.sort(w)

    @testing.for_all_dtypes(no_float16=True, no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_eigvals_sym(self, xp, dtype):
        if not cusolver.check_availability('geev'):
            pytest.skip('geev is not available')
        a = xp.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], dtype)
        a = _get_hermitian(xp, a, 'U')
        w = xp.linalg.eigvals(a)
        w = _real_to_complex(w)
        # Canonicalize the order
        return xp.sort(w)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_eigvals_complex(self, xp, dtype):
        if not cusolver.check_availability('geev'):
            pytest.skip('geev is not available')
        a = xp.array([[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]], dtype)
        w = xp.linalg.eigvals(a)
        # Canonicalize the order
        return xp.sort(w)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_eigvals_hermitian(self, xp, dtype):
        if not cusolver.check_availability('geev'):
            pytest.skip('geev is not available')
        a = xp.array([[1, 2j, 3], [4j, 5, 6j], [7, 8j, 9]], dtype)
        a = _get_hermitian(xp, a, 'U')
        w = xp.linalg.eigvals(a)
        # Canonicalize the order
        return xp.sort(w)


@pytest.mark.parametrize('UPLO', ['U', 'L'])
@pytest.mark.parametrize('shape', [
    (0, 0),
    (2, 0, 0),
    (0, 3, 3),
])
@pytest.mark.skipif(
    runtime.is_hip and driver.get_build_version() < 402,
    reason='eigensolver not added until ROCm 4.2.0')
class TestSymEigenvalueEmpty:

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


@pytest.mark.parametrize(
    "shape",
    [
        (0, 0),
        (2, 0, 0),
        (0, 3, 3),
    ],
)
@pytest.mark.skipif(runtime.is_hip, reason="hip does not support eig")
@pytest.mark.skipif(
    cupy.cuda.runtime.runtimeGetVersion() < 12060,
    reason='Requires CUDA 12.6+')
class TestEigenvalueEmpty:

    @testing.for_dtypes('ifdFD')
    @testing.numpy_cupy_allclose()
    def test_eig(self, xp, dtype, shape):
        if not cusolver.check_availability('geev'):
            pytest.skip('geev is not available')
        a = xp.empty(shape, dtype)
        assert a.size == 0
        return xp.linalg.eig(a)

    @testing.for_dtypes('ifdFD')
    @testing.numpy_cupy_allclose()
    def test_eigvals(self, xp, dtype, shape):
        if not cusolver.check_availability('geev'):
            pytest.skip('geev is not available')
        a = xp.empty(shape, dtype)
        assert a.size == 0
        return xp.linalg.eigvals(a)


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
class TestSymEigenvalueInvalid:

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


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (3,),
        (2, 3),
        (4, 0),
        (2, 2, 3),
        (0, 2, 3),
    ],
)
@pytest.mark.skipif(runtime.is_hip, reason="hip does not support eig")
@pytest.mark.skipif(
    cupy.cuda.runtime.runtimeGetVersion() < 12060,
    reason='Requires CUDA 12.6+')
class TestEigenvalueInvalid:

    def test_eig_shape_error(self, shape):
        if not cusolver.check_availability('geev'):
            pytest.skip('geev is not available')
        for xp in (numpy, cupy):
            a = xp.zeros(shape)
            with pytest.raises(numpy.linalg.LinAlgError):
                xp.linalg.eig(a)

    def test_eigvals_shape_error(self, shape):
        if not cusolver.check_availability('geev'):
            pytest.skip('geev is not available')
        for xp in (numpy, cupy):
            a = xp.zeros(shape)
            with pytest.raises(numpy.linalg.LinAlgError):
                xp.linalg.eigvals(a)


@pytest.mark.skipif(runtime.is_hip, reason="hip does not support eig")
@pytest.mark.skipif(
    cupy.cuda.runtime.runtimeGetVersion() < 12060,
    reason='Requires CUDA 12.6+')
class TestStackedEigenvalues:

    def check_eig(self, ew_gpu, ew_cpu, ev_gpu=None, ev_cpu=None):
        # upcast real to complex -- cupy does always return complex type
        if numpy.isrealobj(ew_cpu) and numpy.iscomplexobj(ew_gpu):
            ew_cpu = ew_cpu.astype(ew_gpu.dtype)
            if ev_gpu is not None:
                ev_cpu = ev_cpu.astype(ev_gpu.dtype)

        # sort by eigenvalues
        ew_cpu_ind = numpy.argsort(ew_cpu, axis=-1)
        ew_gpu_ind = cupy.argsort(ew_gpu, axis=-1)

        ew_cpu = numpy.take_along_axis(ew_cpu, ew_cpu_ind, axis=-1)
        ew_gpu = cupy.take_along_axis(ew_gpu, ew_gpu_ind, axis=-1)
        cupy.testing.assert_allclose(ew_gpu, ew_cpu, rtol=1e-5, atol=1e-4)

        if ev_gpu is not None or ev_cpu is not None:
            ev_cpu = numpy.take_along_axis(
                ev_cpu, ew_cpu_ind[..., None], axis=-1)
            ev_gpu = cupy.take_along_axis(
                ev_gpu, ew_gpu_ind[..., None], axis=-1)

            # eigenvectors can be off by a factor of -1
            scale_vec = numpy.divide(ev_cpu[..., 0], ev_gpu.get()[..., 0])
            ev_cpu *= scale_vec[..., None]

            cupy.testing.assert_allclose(ev_gpu, ev_cpu, rtol=1e-5, atol=1e-4)

    def check_eigvals_for_shape(self, dtype, shape):
        array = testing.shaped_random(shape, numpy, dtype=dtype, seed=42)
        a_cpu = numpy.asarray(array, dtype=dtype)
        a_gpu = cupy.asarray(array, dtype=dtype)
        ew_cpu = numpy.linalg.eigvals(a_cpu)
        ew_gpu = cupy.linalg.eigvals(a_gpu)

        # Check if the input matrix is not broken
        cupy.testing.assert_allclose(a_gpu, a_cpu)

        self.check_eig(ew_gpu, ew_cpu)

    def check_eig_for_shape(self, dtype, shape):
        array = testing.shaped_random(shape, numpy, dtype=dtype, seed=42)
        a_cpu = numpy.asarray(array, dtype=dtype)
        a_gpu = cupy.asarray(array, dtype=dtype)
        ew_cpu, ev_cpu = numpy.linalg.eig(a_cpu)
        ew_gpu, ev_gpu = cupy.linalg.eig(a_gpu)

        # Check if the input matrix is not broken
        cupy.testing.assert_allclose(a_gpu, a_cpu)

        self.check_eig(ew_gpu, ew_cpu, ev_gpu, ev_cpu)

    @testing.for_dtypes([
        numpy.float32, numpy.float64, numpy.complex64, numpy.complex128,
    ])
    def test_3d_eigvals(self, dtype):
        self.check_eigvals_for_shape(dtype, (12, 1, 1))
        self.check_eigvals_for_shape(dtype, (2, 17, 17))
        self.check_eigvals_for_shape(dtype, (1, 4, 4))
        self.check_eigvals_for_shape(dtype, (33, 3, 3))

    @testing.for_dtypes([
        numpy.float32, numpy.float64, numpy.complex64, numpy.complex128,
    ])
    def test_4d_eigvals(self, dtype):
        self.check_eigvals_for_shape(dtype, (2, 7, 4, 4))
        self.check_eigvals_for_shape(dtype, (1, 2, 3, 3))
        self.check_eigvals_for_shape(dtype, (4, 1, 3, 3))
        self.check_eigvals_for_shape(dtype, (6, 4, 7, 7))
        self.check_eigvals_for_shape(dtype, (3, 2, 1, 1))

    @testing.for_dtypes([
        numpy.float32, numpy.float64, numpy.complex64, numpy.complex128,
    ])
    def test_5d_eigvals(self, dtype):
        self.check_eigvals_for_shape(dtype, (2, 7, 3, 4, 4))
        self.check_eigvals_for_shape(dtype, (1, 3, 2, 3, 3))
        self.check_eigvals_for_shape(dtype, (4, 1, 4, 3, 3))
        self.check_eigvals_for_shape(dtype, (6, 4, 1, 7, 7))
        self.check_eigvals_for_shape(dtype, (5, 3, 2, 1, 1))

    @testing.for_dtypes([
        numpy.float32, numpy.float64, numpy.complex64, numpy.complex128,
    ])
    def test_3d_eig(self, dtype):
        self.check_eig_for_shape(dtype, (12, 1, 1))
        self.check_eig_for_shape(dtype, (2, 17, 17))
        self.check_eig_for_shape(dtype, (1, 4, 4))
        self.check_eig_for_shape(dtype, (33, 3, 3))

    @testing.for_dtypes([
        numpy.float32, numpy.float64, numpy.complex64, numpy.complex128,
    ])
    def test_4d_eig(self, dtype):
        self.check_eig_for_shape(dtype, (2, 7, 4, 4))
        self.check_eig_for_shape(dtype, (1, 2, 3, 3))
        self.check_eig_for_shape(dtype, (4, 1, 3, 3))
        self.check_eig_for_shape(dtype, (6, 4, 7, 7))
        self.check_eig_for_shape(dtype, (3, 2, 1, 1))

    @testing.for_dtypes([
        numpy.float32, numpy.float64, numpy.complex64, numpy.complex128,
    ])
    def test_5d_eig(self, dtype):
        self.check_eig_for_shape(dtype, (2, 7, 3, 4, 4))
        self.check_eig_for_shape(dtype, (1, 3, 2, 3, 3))
        self.check_eig_for_shape(dtype, (4, 1, 4, 3, 3))
        self.check_eig_for_shape(dtype, (6, 4, 1, 7, 7))
        self.check_eig_for_shape(dtype, (5, 3, 2, 1, 1))

    @testing.for_dtypes([
        numpy.float32, numpy.float64,
    ])
    def test_real_to_real(self, dtype):
        # add input matrices with real result
        mat1 = numpy.eye(9, dtype=dtype).reshape((1, 9, 9)) * 2
        mat2 = numpy.eye(9, dtype=dtype).reshape((1, 9, 9)) * 3
        a_cpu = numpy.append(mat1, mat2, axis=0)
        a_gpu = cupy.asarray(a_cpu, dtype=dtype)
        ew_cpu, ev_cpu = numpy.linalg.eig(a_cpu)
        ew_gpu, ev_gpu = cupy.linalg.eig(a_gpu)

        # Check if the input matrix is not broken
        cupy.testing.assert_allclose(a_gpu, a_cpu)

        self.check_eig(ew_gpu, ew_cpu, ev_gpu, ev_cpu)

    @testing.for_dtypes([
        numpy.float32, numpy.float64,
    ])
    def test_mixed_real_complex(self, dtype):
        array = testing.shaped_random((7, 9, 9), numpy, dtype=dtype, seed=42)
        a_cpu = numpy.asarray(array, dtype=dtype)

        # add input matrix with real result
        a_cpu = numpy.append(a_cpu, numpy.eye(
            9, dtype=dtype).reshape((1, 9, 9)), axis=0)
        a_gpu = cupy.asarray(a_cpu, dtype=dtype)
        ew_cpu, ev_cpu = numpy.linalg.eig(a_cpu)
        ew_gpu, ev_gpu = cupy.linalg.eig(a_gpu)

        # Check if the input matrix is not broken
        cupy.testing.assert_allclose(a_gpu, a_cpu)

        self.check_eig(ew_gpu, ew_cpu, ev_gpu, ev_cpu)
