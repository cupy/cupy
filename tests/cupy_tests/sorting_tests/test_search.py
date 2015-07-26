import unittest

from cupy import testing


@testing.gpu
class TestSearch(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_all(self, xpy, dtype):
        a = testing.shaped_arange((2, 3), xpy, dtype)
        return a.argmax()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_axis(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype)
        return a.argmax(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_all(self, xpy, dtype):
        a = testing.shaped_arange((2, 3), xpy, dtype)
        return a.argmin()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_prod_axis(self, xpy, dtype):
        a = testing.shaped_arange((2, 3, 4), xpy, dtype)
        return a.argmin(axis=1)
