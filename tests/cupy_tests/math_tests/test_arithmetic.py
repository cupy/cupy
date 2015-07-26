import unittest

import numpy

from cupy import testing


@testing.gpu
class TestArithmetic(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary(self, name, xpy, dtype):
        a = testing.shaped_arange((2, 3), xpy, dtype)
        return getattr(xpy, name)(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_binary(self, name, xpy, dtype):
        a = testing.shaped_arange((2, 3), xpy, dtype)
        b = testing.shaped_reverse_arange((2, 3), xpy, dtype)
        return getattr(xpy, name)(a, b)

    @testing.for_dtypes(['?', 'b', 'h', 'i', 'q', 'e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_negative(self, name, xpy, dtype):
        a = xpy.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        return getattr(xpy, name)(a)

    @testing.for_dtypes(['?', 'b', 'h', 'i', 'q', 'e', 'f', 'd'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_binary_negative(self, name, xpy, dtype):
        a = xpy.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        b = xpy.array([4, 3, 2, 1, -1, -2], dtype=dtype)
        return getattr(xpy, name)(a, b)

    def test_add(self):
        self.check_binary('add')

    def test_reciprocal(self):
        numpy.seterr(divide='ignore', invalid='ignore')
        self.check_unary('reciprocal')

    def test_multiply(self):
        self.check_binary('multiply')

    def test_divide(self):
        numpy.seterr(divide='ignore')
        self.check_binary('divide')

    def test_divide_negative(self):
        numpy.seterr(divide='ignore')
        self.check_binary_negative('divide')

    def test_power(self):
        self.check_binary('power')

    def test_power_negative(self):
        self.check_binary_negative('power')

    def test_subtract(self):
        self.check_binary('subtract')

    def test_true_divide(self):
        numpy.seterr(divide='ignore')
        self.check_binary('true_divide')

    def test_true_divide_negative(self):
        numpy.seterr(divide='ignore')
        self.check_binary_negative('true_divide')

    def test_floor_divide(self):
        numpy.seterr(divide='ignore')
        self.check_binary('floor_divide')

    def test_floor_divide_negative(self):
        numpy.seterr(divide='ignore')
        self.check_binary_negative('floor_divide')

    def test_fmod(self):
        numpy.seterr(divide='ignore')
        self.check_binary('fmod')

    def test_fmod_negative(self):
        numpy.seterr(divide='ignore')
        self.check_binary_negative('fmod')

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_modf(self, xpy, dtype):
        a = xpy.array([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], dtype=dtype)
        b, c = xpy.modf(a)
        d = xpy.empty((2, 7), dtype=dtype)
        d[0] = b
        d[1] = c
        return d

    def test_remainder(self):
        numpy.seterr(divide='ignore')
        self.check_binary('remainder')

    def test_remainder_negative(self):
        numpy.seterr(divide='ignore')
        self.check_binary_negative('remainder')
