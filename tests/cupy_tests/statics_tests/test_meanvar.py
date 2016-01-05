import unittest

from cupy import testing


@testing.gpu
class TestMeanVar(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_mean_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return a.mean()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_mean_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.mean(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_mean_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.mean(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_mean_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.mean(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_var_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return a.var()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_var_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.var(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_var_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.var(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_var_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.var(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_std_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return a.std()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_std_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.std(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_std_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.std(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_std_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.std(a, axis=1)
