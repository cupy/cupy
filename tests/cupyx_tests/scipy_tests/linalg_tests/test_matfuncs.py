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


class TestExpM:

    def test_zero(self):
        a = cupy.array([[0., 0], [0, 0]])
        assert cupy.abs(cx_linalg.expm(a) - cupy.eye(2)).all() < 1e-10

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
