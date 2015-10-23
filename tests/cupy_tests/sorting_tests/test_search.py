import unittest

from cupy import testing


@testing.gpu
class TestSearch(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return a.argmax()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return a.argmax(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmax(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmax(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmax(axis=2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return a.argmin()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return a.argmin(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmin(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmin(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.argmin(axis=2)


class TestWhere(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_basic(self, xp, dtype):
        cond = testing.shaped_random((2, 3, 4), xp, dtype=xp.bool_)
        x = testing.shaped_random((2, 3, 4), xp, dtype)
        y = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.where(cond, x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_condition_broadcast(self, xp, dtype):
        cond = testing.shaped_random((4,), xp, dtype=xp.bool_)
        x = testing.shaped_random((2, 3, 4), xp, dtype)
        y = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.where(cond, x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_value_broadcast(self, xp, dtype):
        cond = testing.shaped_random((2, 3, 4), xp, dtype=xp.bool_)
        x = testing.shaped_random((2, 3, 4), xp, dtype)
        y = testing.shaped_random((3, 4), xp, dtype)
        return xp.where(cond, x, y)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_condition_value_broadcast(self, xp, dtype):
        cond = testing.shaped_random((3, 4), xp, dtype=xp.bool_)
        x = testing.shaped_random((2, 3, 4), xp, dtype)
        y = testing.shaped_random((4,), xp, dtype)
        return xp.where(cond, x, y)

    @testing.numpy_cupy_raises()
    def test_one_argument(self, xp):
        cond = testing.shaped_random((3, 4), xp, dtype=xp.bool_)
        x = testing.shaped_random((2, 3, 4), xp, xp.int32)
        xp.where(cond, x)
