import unittest

import numpy

from cupy import testing


@testing.gpu
class TestMisc(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary(self, name, xp, dtype, no_bool=False):
        if no_bool and numpy.dtype(dtype).char == '?':
            return numpy.int_(0)
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_binary(self, name, xp, dtype, no_bool=False):
        if no_bool and numpy.dtype(dtype).char == '?':
            return numpy.int_(0)
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a, b)

    @testing.for_dtypes(['?', 'b', 'h', 'i', 'q', 'e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_negative(self, name, xp, dtype, no_bool=False):
        if no_bool and numpy.dtype(dtype).char == '?':
            return numpy.int_(0)
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        return getattr(xp, name)(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_array_equal()
    def check_binary_nan(self, name, xp, dtype):
        a = xp.array([-3, numpy.NAN, -1, numpy.NAN, 0, numpy.NAN, 2],
                     dtype=dtype)
        b = xp.array([numpy.NAN, numpy.NAN, 1, 0, numpy.NAN, -1, -2],
                     dtype=dtype)
        return getattr(xp, name)(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_clip1(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.clip(3, 13)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_external_clip(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.clip(a, 3, 13)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_clip2(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        a_min = xp.array([3, 4, 5, 6], dtype=dtype)
        a_max = xp.array([[10], [9], [8]], dtype=dtype)
        return a.clip(a_min, a_max)

    @testing.with_requires('numpy>=1.11.2')
    def test_sqrt(self):
        # numpy.sqrt is broken in numpy<1.11.2
        self.check_unary('sqrt')

    def test_square(self):
        self.check_unary('square')

    def test_absolute(self):
        self.check_unary('absolute')

    def test_absolute_negative(self):
        self.check_unary_negative('absolute')

    def test_sign(self):
        self.check_unary('sign', no_bool=True)

    def test_sign_negative(self):
        self.check_unary_negative('sign', no_bool=True)

    def test_maximum(self):
        self.check_binary('maximum')

    def test_maximum_nan(self):
        self.check_binary_nan('maximum')

    def test_minimum(self):
        self.check_binary('minimum')

    def test_minimum_nan(self):
        self.check_binary_nan('minimum')

    def test_fmax(self):
        self.check_binary('fmax')

    def test_fmax_nan(self):
        self.check_binary_nan('fmax')

    def test_fmin(self):
        self.check_binary('fmin')

    def test_fmin_nan(self):
        self.check_binary_nan('fmin')
