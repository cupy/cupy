import unittest

from cupy import testing


@testing.gpu
class TestCorrelation(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_corrcoef(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.corrcoef(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_corrcoef_diag_exception(self, xp, dtype):
        a = testing.shaped_arange((1, 3), xp, dtype)
        return xp.corrcoef(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_corrcoef_y(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        y = testing.shaped_arange((2, 3), xp, dtype)
        return xp.corrcoef(a, y=y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_corrcoef_rowvar(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        y = testing.shaped_arange((2, 3), xp, dtype)
        return xp.corrcoef(a, y=y, rowvar=False)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cov(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.cov(a)

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cov_empty(self, xp, dtype):
        a = testing.shaped_arange((0, 1), xp, dtype)
        return xp.cov(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cov_y(self, xp, dtype):
        a = testing.shaped_arange((2,), xp, dtype)
        y = testing.shaped_arange((2,), xp, dtype)
        return xp.cov(a, y=y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cov_rowvar(self, xp, dtype):
        a = testing.shaped_arange((1, 3), xp, dtype)
        y = testing.shaped_arange((1, 3), xp, dtype)
        return xp.cov(a, y=y, rowvar=False)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cov_rowvar_shape(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        y = testing.shaped_arange((2, 3), xp, dtype)
        return xp.cov(a, y=y, rowvar=False)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cov_bias(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.cov(a, bias=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cov_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.cov(a, ddof=2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cov_negative_degrees_of_freedom(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        with testing.assert_warns(RuntimeWarning):
            return xp.cov(a, ddof=4)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_cov_invalid_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        xp.cov(a, ddof=1.2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_cov_too_much_ndim(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 2), xp, dtype)
        xp.cov(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_cov_y_too_much_ndim(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        y = testing.shaped_arange((3, 4, 2), xp, dtype)
        xp.cov(a, y=y)
