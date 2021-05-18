import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.parameterize(*testing.product({
    'trim': [True, False]
}))
@testing.gpu
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
        a = [xp.array([3, 5, 7, -4, 1, 2], dtype)]
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


@testing.gpu
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


@testing.gpu
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


@testing.gpu
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
