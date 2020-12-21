import numpy
import unittest

import cupy
from cupy import testing


@testing.gpu
class TestNdim(unittest.TestCase):

    @testing.numpy_cupy_equal()
    def test_ndim_ndarray1d(self, xp):
        return xp.ndim(xp.arange(5))

    @testing.numpy_cupy_equal()
    def test_ndim_ndarray2d(self, xp):
        return xp.ndim(xp.ones((2, 4)))

    @testing.numpy_cupy_equal()
    def test_ndim_ndarray0d(self, xp):
        return xp.ndim(xp.asarray(5))

    @testing.numpy_cupy_equal()
    def test_ndim_scalar(self, xp):
        return xp.ndim(5)

    @testing.numpy_cupy_equal()
    def test_ndim_none(self, xp):
        return xp.ndim(None)

    @testing.numpy_cupy_equal()
    def test_ndim_string(self, xp):
        return xp.ndim('abc')

    @testing.numpy_cupy_equal()
    def test_ndim_list1(self, xp):
        return xp.ndim([1, 2, 3])

    @testing.numpy_cupy_equal()
    def test_ndim_list2(self, xp):
        return xp.ndim([[1, 2, 3], [4, 5, 6]])

    @testing.numpy_cupy_equal()
    def test_ndim_tuple(self, xp):
        return xp.ndim(((1, 2, 3), (4, 5, 6)))

    @testing.numpy_cupy_equal()
    def test_ndim_set(self, xp):
        return xp.ndim({1, 2, 3})

    @testing.numpy_cupy_equal()
    def test_ndim_object(self, xp):
        return xp.ndim(dict(a=5, b='b'))

    # numpy.dim works on CuPy arrays and cupy.ndim works on NumPy arrays
    def test_ndim_array_function(self):
        a = cupy.ones((4, 4))
        assert numpy.ndim(a) == 2

        a = cupy.asarray(5)
        assert numpy.ndim(a) == 0

        a = numpy.ones((4, 4))
        assert cupy.ndim(a) == 2

        a = numpy.asarray(5)
        assert cupy.ndim(a) == 0
