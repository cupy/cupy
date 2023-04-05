import numpy as np
from numpy import linalg
from cupyx.scipy.linalg import bandwidth
from pytest import raises
from cupy import testing


class TestBadwidth:
    @testing.for_all_dtypes(no_complex=True)
    def test_bandwidth_non2d_input(self, dtype):
        A = np.array([1, 2, 3], dtype=dtype)
        raises(linalg.LinAlgError, bandwidth, A)
        A = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=dtype)
        raises(linalg.LinAlgError, bandwidth, A)

    @testing.for_all_dtypes(no_complex=True)
    def test_bandwidth_square_symmetric_inputs(self, dtype):
        n = 10
        for k in range(1, 10):
            R = np.zeros([n, n], dtype=dtype)
            # form a banded matrix inplace
            R[[x for x in range(n)], [x for x in range(n)]] = 1
            R[[x for x in range(n - k)], [x for x in range(k, n)]] = 1
            R[[x for x in range(1, n)], [x for x in range(n - 1)]] = 1
            R[[x for x in range(k, n)], [x for x in range(n - k)]] = 1
            testing.assert_array_equal(bandwidth(R), (k, k))

    @testing.for_all_dtypes(no_complex=True)
    def test_bandwidth_square_asymmetric_inputs_c(self, dtype):
        n = 20
        a = 5
        b = 4
        R = np.zeros([n, n], dtype=dtype)
        # form a banded matrix inplace
        R[[x for x in range(n)], [x for x in range(n)]] = 1
        R[[x for x in range(a, n)], [x for x in range(n - a)]] = 1
        R[[x for x in range(n - b)], [x for x in range(b, n)]] = 1
        testing.assert_array_equal(bandwidth(R), (a, b))

    @testing.for_all_dtypes(no_complex=True)
    def test_bandwidth_square_asymmetric_inputs_f(self, dtype):
        n = 20
        a = 5
        b = 4
        R = np.zeros([n, n], dtype=dtype, order='F')
        # form a banded matrix inplace
        R[[x for x in range(n)], [x for x in range(n)]] = 1
        R[[x for x in range(a, n)], [x for x in range(n - a)]] = 1
        R[[x for x in range(n - b)], [x for x in range(b, n)]] = 1
        testing.assert_array_equal(bandwidth(R), (a, b))

    @testing.for_all_dtypes(no_complex=True)
    def test_bandwidth_square_inputs_f(self, dtype):
        n = 10
        for k in range(1, 10):
            R = np.zeros([n, n], dtype=dtype, order='F')
            # form a banded matrix inplace
            R[[x for x in range(n)], [x for x in range(n)]] = 1
            R[[x for x in range(n - k)], [x for x in range(k, n)]] = 1
            R[[x for x in range(1, n)], [x for x in range(n - 1)]] = 1
            R[[x for x in range(k, n)], [x for x in range(n - k)]] = 1
            testing.assert_array_equal(bandwidth(R), (k, k))

    @testing.for_all_dtypes(no_complex=True)
    def test_bandwidth_rect_inputs_c(self, dtype):
        n, m = 10, 20
        k = 5
        R = np.zeros([n, m], dtype=dtype)
        # form a banded matrix inplace
        R[[x for x in range(n)], [x for x in range(n)]] = 1
        R[[x for x in range(n - k)], [x for x in range(k, n)]] = 1
        R[[x for x in range(1, n)], [x for x in range(n - 1)]] = 1
        R[[x for x in range(k, n)], [x for x in range(n - k)]] = 1
        testing.assert_array_equal(bandwidth(R), (k, k))

    @testing.for_all_dtypes(no_complex=True)
    def test_bandwidth_rect_inputs_f(self, dtype):
        n, m = 10, 20
        k = 5
        R = np.zeros([n, m], dtype=dtype, order='F')
        # form a banded matrix inplace
        R[[x for x in range(n)], [x for x in range(n)]] = 1
        R[[x for x in range(n - k)], [x for x in range(k, n)]] = 1
        R[[x for x in range(1, n)], [x for x in range(n - 1)]] = 1
        R[[x for x in range(k, n)], [x for x in range(n - k)]] = 1
        testing.assert_array_equal(bandwidth(R), (k, k))
