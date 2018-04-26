import unittest

from cupy import testing


@testing.gpu
class TestCorrelation(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cov(self, xp, dtype):
        m = testing.shaped_random((2, 3), xp, dtype)
        return xp.cov(m)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cov_y(self, xp, dtype):
        m = testing.shaped_random((2, 3), xp, dtype)
        y = testing.shaped_random((2, 3), xp, dtype)
        return xp.cov(m, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cov_rowvar(self, xp, dtype):
        m = testing.shaped_random((2, 3), xp, dtype)
        y = testing.shaped_random((2, 3), xp, dtype)
        return xp.cov(m, y, rowvar=False)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cov_bias(self, xp, dtype):
        m = testing.shaped_random((2, 3), xp, dtype)
        y = testing.shaped_random((2, 3), xp, dtype)
        return xp.cov(m, y, bias=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cov_ddof(self, xp, dtype):
        m = testing.shaped_random((2, 3), xp, dtype)
        y = testing.shaped_random((2, 3), xp, dtype)
        return xp.cov(m, y, ddof=2)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cov_fweights(self, xp, dtype):
        m = testing.shaped_random((2, 3), xp, dtype)
        fweights = testing.shaped_arange((3,), xp, dtype=int) + 1
        return xp.cov(m, fweights=fweights)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cov_aweights(self, xp, dtype):
        m = testing.shaped_random((2, 3), xp, dtype)
        aweights = testing.shaped_arange((3,), xp, dtype=int) + 1
        return xp.cov(m, aweights=aweights)
