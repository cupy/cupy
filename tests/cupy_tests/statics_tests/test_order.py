import unittest
import warnings

from cupy import testing


@testing.gpu
class TestOrder(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmax_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.nanmax(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmax_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.nanmax(a, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmax_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmax(a, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmax_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmax(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmax_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmax(a, axis=2)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmax_nan(self, xp, dtype):
        a = xp.array([float('nan'), 1, -1], dtype)
        with warnings.catch_warnings():
            return xp.nanmax(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmax_all_nan(self, xp, dtype):
        a = xp.array([float('nan'), float('nan')], dtype)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            m = xp.nanmax(a)
        self.assertEqual(len(w), 1)
        self.assertIs(w[0].category, RuntimeWarning)
        return m

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmin_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.nanmin(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmin_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.nanmin(a, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmin_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmin(a, axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmin_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmin(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmin_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.nanmin(a, axis=2)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmin_nan(self, xp, dtype):
        a = xp.array([float('nan'), 1, -1], dtype)
        with warnings.catch_warnings():
            return xp.nanmin(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanmin_all_nan(self, xp, dtype):
        a = xp.array([float('nan'), float('nan')], dtype)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            m = xp.nanmin(a)
        self.assertEqual(len(w), 1)
        self.assertIs(w[0].category, RuntimeWarning)
        return m
