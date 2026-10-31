from __future__ import annotations

import numpy as np
from numpy import linalg

import cupy
from cupyx.scipy.linalg import khatri_rao
from cupyx.scipy import linalg as cx_linalg
import pytest
from cupy import testing

try:
    import scipy.linalg    # noqa
except ImportError:
    pass


class TestKhatriRao:
    @testing.for_all_dtypes()
    def test_basic(self, dtype):
        A = np.array([[1, 2], [3, 4]], dtype=dtype)
        B = np.array([[5, 6], [7, 8]], dtype=dtype)
        prod = np.array([[5, 12],
                         [7, 16],
                         [15, 24],
                         [21, 32]], dtype=dtype)

        testing.assert_array_equal(khatri_rao(A, B), prod)

    @testing.for_all_dtypes()
    def test_shape(self, dtype):
        M = khatri_rao(np.empty([2, 2], dtype=dtype),
                       np.empty([2, 2], dtype=dtype))
        testing.assert_array_equal(M.shape, (4, 2))

    @testing.for_all_dtypes()
    def test_number_of_columns_equality(self, dtype):
        with pytest.raises(ValueError):
            A = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
            B = np.array([[1, 2], [3, 4]], dtype=dtype)
            khatri_rao(A, B)

    @testing.for_all_dtypes()
    def test_to_assure_2d_array(self, dtype):
        with pytest.raises(linalg.LinAlgError):
            # both arrays are 1-D
            A = np.array([1, 2, 3], dtype=dtype)
            B = np.array([4, 5, 6], dtype=dtype)
            khatri_rao(A, B)

        with pytest.raises(linalg.LinAlgError):
            # first array is 1-D
            A = np.array([1, 2, 3], dtype=dtype)
            B = np.array([
                [1, 2, 3],
                [4, 5, 6]
            ], dtype=dtype)
            khatri_rao(A, B)

        with pytest.raises(linalg.LinAlgError):
            # first array is 1-D
            A = np.array([
                [1, 2, 3],
                [4, 5, 6]
            ], dtype=dtype)
            B = np.array([1, 2, 3], dtype=dtype)
            khatri_rao(A, B)

    @testing.for_all_dtypes()
    def test_equality_of_two_equations(self, dtype):
        A = np.array([[1, 2], [3, 4]], dtype=dtype)
        B = np.array([[5, 6], [7, 8]], dtype=dtype)

        res1 = khatri_rao(A, B)
        res2 = np.vstack([np.kron(A[:, k], B[:, k])
                          for k in range(B.shape[1])]).T

        testing.assert_array_equal(res1, res2)


@testing.with_requires("scipy")
class TestExpM:

    def test_zero(self):
        a = cupy.array([[0., 0], [0, 0]])
        testing.assert_allclose(cx_linalg.expm(a), cupy.eye(2),
                                rtol=1e-14, atol=1e-15)

    def test_empty_matrix_input(self):
        # handle gh-11082
        A = np.zeros((0, 0))
        result = cx_linalg.expm(A)
        assert result.size == 0

    @testing.numpy_cupy_allclose(scipy_name='scp', contiguous_check=False)
    def test_2x2_input(self, xp, scp):
        a = xp.array([[1, 4], [1, 1]])
        return scp.linalg.expm(a)

    @pytest.mark.parametrize('a', ([[1, 4], [1, 1]],
                                   [[1, 3], [1, -1]],
                                   [[1, 3], [4, 5]],
                                   [[1, 3], [5, 3]],
                                   [[4, 5], [-3, -4]])
                             )
    @testing.numpy_cupy_allclose(scipy_name='scp', contiguous_check=False)
    def test_nx2x2_input(self, xp, scp, a):
        a = xp.asarray(a)
        return scp.linalg.expm(a)

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(scipy_name='scp', contiguous_check=False)
    def test_dtypes(self, xp, scp, dtype):
        a = xp.eye(2, dtype=dtype)
        return scp.linalg.expm(a)

    @pytest.mark.parametrize('norm', [
        1.495585217958292e-2, 2.539398330063230e-1,
        9.504178996162932e-1, 2.097847961257068,
        5.371920351148152, 2 * 5.371920351148152])
    @pytest.mark.parametrize('factor', [0.99, 1., 1.01])
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(
        scipy_name='scp', contiguous_check=False,
        rtol={'default': 1e-12, np.float32: 1e-5, np.complex64: 1e-5},
        atol={'default': 1e-14, np.float32: 1e-6, np.complex64: 1e-6})
    def test_pade_thresholds(self, xp, scp, dtype, norm, factor):
        a = xp.asarray([[0, norm * factor], [-norm * factor, 0]],
                       dtype=dtype)
        return scp.linalg.expm(a)

    @pytest.mark.parametrize('norm', [0.01, 0.1, 0.5, 1.5, 3., 8.])
    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(
        scipy_name='scp', contiguous_check=False,
        rtol={'default': 1e-12, np.float32: 2e-5, np.complex64: 2e-5},
        atol={'default': 1e-14, np.float32: 2e-6, np.complex64: 2e-6})
    def test_random_noncontiguous(self, xp, scp, dtype, norm):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((8, 8))
        if np.issubdtype(dtype, np.complexfloating):
            a = a + 1j * rng.standard_normal((8, 8))
        a -= np.eye(8) * np.trace(a) / 8
        a *= norm / np.linalg.norm(a, 1)
        a += 0.5 * np.eye(8)
        a = xp.asarray(a, dtype=dtype)[::-1, ::-1]
        return scp.linalg.expm(a)

    @testing.for_dtypes('fdFD')
    @testing.numpy_cupy_allclose(
        scipy_name='scp', contiguous_check=False,
        rtol={'default': 1e-12, np.float32: 5e-5, np.complex64: 5e-5},
        atol={'default': 1e-13, np.float32: 5e-6, np.complex64: 5e-6})
    def test_large_skew_hermitian(self, xp, scp, dtype):
        rng = np.random.default_rng(123)
        a = rng.standard_normal((8, 8))
        if np.issubdtype(dtype, np.complexfloating):
            a = a + 1j * rng.standard_normal((8, 8))
        a = a - a.conj().T
        a *= 100 / np.linalg.norm(a, 1)
        return scp.linalg.expm(xp.asarray(a, dtype=dtype))

    @pytest.mark.parametrize('scale', [0.01, 0.1, 0.5, 1.5, 3., 8., 100.])
    @testing.for_dtypes('fdFD')
    def test_nilpotent(self, dtype, scale):
        a = cupy.asarray([[0, scale], [0, 0]], dtype=dtype)
        # A**2 = 0, so the exponential is exactly I + A.
        tol = 1e-6 if dtype in (np.float32, np.complex64) else 1e-13
        testing.assert_allclose(cx_linalg.expm(a), cupy.eye(2) + a,
                                rtol=tol, atol=tol)

    def test_input_unchanged(self):
        a = testing.shaped_random((4, 4), cupy, dtype=np.float64)
        original = a.copy()
        cx_linalg.expm(a)
        testing.assert_array_equal(a, original)

    def test_stream(self):
        with cupy.cuda.Stream(non_blocking=True) as stream:
            a = cupy.asarray([[0., 0.1], [0., 0.]])
            result = cx_linalg.expm(a)
            expected = cupy.eye(2) + a
        stream.synchronize()
        testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-15)

    @testing.numpy_cupy_allclose(scipy_name='scp', contiguous_check=False)
    def test_gh_9138(self, xp, scp):
        rng = np.random.default_rng(123)
        a = rng.standard_normal(size=(3, 3))
        a = a + 1j*rng.standard_normal(size=(3, 3))
        A = a.conj().T - a
        A = xp.asarray(A)
        return scp.linalg.expm(A)

    def test_gh_9138_2(self):
        # make sure expm(antisym matrix) is unitary (to a good accuracy)
        rng = np.random.default_rng(123)
        a = rng.standard_normal(size=(3, 3))
        a = a + 1j*rng.standard_normal(size=(3, 3))
        A = a.conj().T - a
        A = cupy.asarray(A)

        U = cx_linalg.expm(A)
        testing.assert_allclose(U.conj().T @ U, cupy.eye(3), atol=1e-14)
