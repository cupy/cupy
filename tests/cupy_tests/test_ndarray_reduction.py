import unittest

from cupy import testing


@testing.gpu
class TestArrayReduction(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_all(self, xpy, dtype):
        a = testing.shaped_random((2, 3), xpy, dtype)
        return a.max()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_axis_large(self, xpy, dtype):
        a = testing.shaped_random((3, 1000), xpy, dtype)
        return a.max(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_axis0(self, xpy, dtype):
        a = testing.shaped_random((2, 3, 4), xpy, dtype)
        return a.max(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_axis1(self, xpy, dtype):
        a = testing.shaped_random((2, 3, 4), xpy, dtype)
        return a.max(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_axis2(self, xpy, dtype):
        a = testing.shaped_random((2, 3, 4), xpy, dtype)
        return a.max(axis=2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_all(self, xpy, dtype):
        a = testing.shaped_random((2, 3), xpy, dtype)
        return a.min()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_axis_large(self, xpy, dtype):
        a = testing.shaped_random((3, 1000), xpy, dtype)
        return a.min(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_axis0(self, xpy, dtype):
        a = testing.shaped_random((2, 3, 4), xpy, dtype)
        return a.min(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_axis1(self, xpy, dtype):
        a = testing.shaped_random((2, 3, 4), xpy, dtype)
        return a.min(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_axis2(self, xpy, dtype):
        a = testing.shaped_random((2, 3, 4), xpy, dtype)
        return a.min(axis=2)
