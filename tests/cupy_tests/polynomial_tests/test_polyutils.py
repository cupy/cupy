import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.parameterize(*testing.product({
    'trim': [True, False]
}))
class TestAsSeries(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_as_series(self, xp, dtype):
        a = testing.shaped_random((1000,), xp, dtype)
        return xp.polynomial.polyutils.as_series(a, trim=self.trim)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_as_series_zeros(self, xp, dtype):
        a = xp.zeros(10, dtype)
        return xp.polynomial.polyutils.as_series(a, trim=self.trim)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_as_series_trailing_zeros(self, xp, dtype):
        a = xp.array([3, 5, 7, 0, 0, 0], dtype)
        return xp.polynomial.polyutils.as_series(a, trim=self.trim)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_as_series_list(self, xp, dtype):
        a = [xp.array([3, 5, 7, -4, 1, 2]).astype(dtype)]
        return xp.polynomial.polyutils.as_series(a, trim=self.trim)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_as_series_2dim(self, xp, dtype):
        a = testing.shaped_random((4, 5), xp, dtype)
        return xp.polynomial.polyutils.as_series(a, trim=self.trim)

    @testing.for_all_dtypes()
    def test_as_series_ndim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((3, 4, 5), xp, dtype)
            with pytest.raises(ValueError):
                xp.polynomial.polyutils.as_series(a, trim=self.trim)

    def test_as_series_nocommon_types(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((5,), xp, dtype=bool)
            with pytest.raises(ValueError):
                xp.polynomial.polyutils.as_series(a, trim=self.trim)


class TestTrimseq(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trimseq(self, xp, dtype):
        a = testing.shaped_random((1000,), xp, dtype)
        return xp.polynomial.polyutils.trimseq(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trimseq_zeros(self, xp, dtype):
        a = xp.zeros(10, dtype)
        b = xp.polynomial.polyutils.trimseq(a)
        assert xp.shares_memory(a[:1], b) is True
        return b

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trimseq_trailing_zeros(self, xp, dtype):
        a = xp.array([1, 5, 2, 0, 0, 0], dtype)
        return xp.polynomial.polyutils.trimseq(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_trimseq_zerosize(self, xp, dtype):
        a = xp.zeros((0,), dtype)
        return xp.polynomial.polyutils.trimseq(a)

    @testing.for_all_dtypes()
    def test_trimseq_zero_dim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((), xp, dtype)
            with pytest.raises(TypeError):
                xp.polynomial.polyutils.trimseq(a)

    @testing.for_all_dtypes()
    def test_trimseq_ndim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((3, 4, 5), xp, dtype)
            with pytest.raises(ValueError):
                xp.polynomial.polyutils.trimseq(a)


@testing.parameterize(*testing.product({
    'tol': [0, 1e-3]
}))
class TestTrimcoef(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_trimcoef(self, xp, dtype):
        a = testing.shaped_random((10,), xp, dtype)
        return xp.polynomial.polyutils.trimcoef(a, dtype(self.tol))

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_trimcoef_trailing_zeros(self, xp, dtype):
        a = xp.array([1, 2, 0, 3, 0, 3e-4, 1e-3], dtype)
        return xp.polynomial.polyutils.trimcoef(a, dtype(self.tol))

    @testing.for_all_dtypes_combination(
        names=['dtype1', 'dtype2'], no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_trimcoef_diff_types(self, xp, dtype1, dtype2):
        a = xp.array([1, 2, 0, 3, 0], dtype1)
        return xp.polynomial.polyutils.trimcoef(a, dtype2(self.tol))

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_trimcoef_zeros(self, xp, dtype):
        a = xp.zeros(10, dtype)
        return xp.polynomial.polyutils.trimcoef(a, dtype(self.tol))

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_trimcoef_almost_zeros(self, xp, dtype):
        a = xp.array([1e-3, 1e-5, 1e-4], dtype)
        return xp.polynomial.polyutils.trimcoef(a, dtype(self.tol))

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_trimcoef_zero_dim(self, xp, dtype):
        a = testing.shaped_random((), xp, dtype)
        return xp.polynomial.polyutils.trimcoef(a, dtype(self.tol))


class TestTrimcoefInvalid(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    def test_trimcoef_neg_tol(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((0,), xp, dtype)
            with pytest.raises(ValueError):
                xp.polynomial.polyutils.trimcoef(a, -1e-3)

    @testing.for_all_dtypes(no_bool=True)
    def test_trimcoef_zero_size(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((0,), xp, dtype)
            with pytest.raises(ValueError):
                xp.polynomial.polyutils.trimcoef(a)

    @testing.for_all_dtypes(no_bool=True)
    def test_trimcoef_ndim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_random((2, 3), xp, dtype)
            with pytest.raises(ValueError):
                xp.polynomial.polyutils.trimcoef(a)

    def test_trimcoef_bool(self):
        for xp in (numpy, cupy):
            a = testing.shaped_random((5,), xp, bool)
            with pytest.raises(ValueError):
                xp.polynomial.polyutils.trimcoef(a)


class TestGetDomain(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_getdomain_real(self, xp, dtype):
        if numpy.dtype(dtype).kind == 'u':  # u for unsigned integers
            x = xp.array([1, 10, 3, 5], dtype=dtype)
        else:
            x = xp.array([1, 10, 3, -1], dtype=dtype)
        return xp.polynomial.polyutils.getdomain(x)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_getdomain_complex(self, xp, dtype):
        if dtype not in (xp.complex64, xp.complex128):
            return xp.array([0], dtype=dtype)
        x = xp.array([1 + 1j, 1 - 1j, 0, 2], dtype=dtype)
        return xp.polynomial.polyutils.getdomain(x)

    @testing.numpy_cupy_allclose()
    def test_getdomain_empty(self, xp):
        x = xp.array([], dtype=xp.float64)
        try:
            return xp.polynomial.polyutils.getdomain(x)
        except ValueError:
            return xp.array([0, 0], dtype=xp.float64)

    def test_getdomain_ndim(self):
        for xp in (numpy, cupy):
            x = xp.array([[1, 2], [3, 4]], dtype=xp.float64)
            with pytest.raises(ValueError):
                xp.polynomial.polyutils.getdomain(x)
