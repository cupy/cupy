import sys
import unittest

import numpy

from cupy import testing


@testing.gpu
class TestMisc(unittest.TestCase):

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

    @testing.for_dtypes(['?', 'b', 'h', 'i', 'q', 'e', 'f', 'd', 'F', 'D'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_negative(self, name, xp, dtype, no_bool=False):
        if no_bool and numpy.dtype(dtype).char == '?':
            return numpy.int_(0)
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        if numpy.dtype(dtype).kind == 'c':
            a += (a * 1j).astype(dtype)
        return getattr(xp, name)(a)

    @testing.for_dtypes(['e', 'f', 'd', 'F', 'D'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_inf(self, name, xp, dtype):
        inf = numpy.inf
        if numpy.dtype(dtype).kind != 'c':
            a = xp.array([0, -1, 1, -inf, inf], dtype=dtype)
        else:
            a = xp.array([complex(x, y)
                          for x in [0, -1, 1, -inf, inf]
                          for y in [0, -1, 1, -inf, inf]],
                         dtype=dtype)
        return getattr(xp, name)(a)

    @testing.for_dtypes(['e', 'f', 'd', 'F', 'D'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_nan(self, name, xp, dtype):
        nan = numpy.nan
        if numpy.dtype(dtype).kind != 'c':
            a = xp.array([0, -1, 1, -nan, nan], dtype=dtype)
        else:
            a = xp.array([complex(x, y)
                          for x in [0, -1, 1, -nan, nan]
                          for y in [0, -1, 1, -nan, nan]],
                         dtype=dtype)
        return getattr(xp, name)(a)

    @testing.for_dtypes(['e', 'f', 'd', 'F', 'D'])
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_inf_nan(self, name, xp, dtype):
        inf = numpy.inf
        nan = numpy.nan
        if numpy.dtype(dtype).kind != 'c':
            a = xp.array([0, -1, 1, -inf, inf, -nan, nan], dtype=dtype)
        else:
            a = xp.array([complex(x, y)
                          for x in [0, -1, 1, -inf, inf, -nan, nan]
                          for y in [0, -1, 1, -inf, inf, -nan, nan]],
                         dtype=dtype)
        return getattr(xp, name)(a)

    @testing.for_dtypes(['e', 'f', 'd', 'F', 'D'])
    @testing.numpy_cupy_array_equal()
    def check_binary_nan(self, name, xp, dtype):
        a = xp.array([-3, numpy.NAN, -1, numpy.NAN, 0, numpy.NAN, 2],
                     dtype=dtype)
        b = xp.array([numpy.NAN, numpy.NAN, 1, 0, numpy.NAN, -1, -2],
                     dtype=dtype)
        return getattr(xp, name)(a, b)

    @unittest.skipIf(
        sys.platform == 'win32', 'dtype problem on Windows')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_clip1(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.clip(3, 13)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_clip3(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.clip(3, 13)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_clip_min_none(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.clip(None, 3)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_clip_max_none(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.clip(3, None)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_raises(accept_error=ValueError)
    def test_clip_min_max_none(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.clip(None, None)

    @unittest.skipIf(
        sys.platform == 'win32', 'dtype problem on Windows')
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_external_clip1(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.clip(a, 3, 13)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_external_clip2(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.clip(a, 3, 13)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_clip2(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        a_min = xp.array([3, 4, 5, 6], dtype=dtype)
        a_max = xp.array([[10], [9], [8]], dtype=dtype)
        return a.clip(a_min, a_max)

    def test_sqrt(self):
        self.check_unary('sqrt')

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_cbrt(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.cbrt(a)

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

    def test_nan_to_num(self):
        self.check_unary('nan_to_num')

    def test_nan_to_num_negative(self):
        self.check_unary_negative('nan_to_num')

    def test_nan_to_num_for_old_numpy(self):
        self.check_unary('nan_to_num', no_bool=True)

    def test_nan_to_num_negative_for_old_numpy(self):
        self.check_unary_negative('nan_to_num', no_bool=True)

    def test_nan_to_num_inf(self):
        self.check_unary_inf('nan_to_num')

    def test_nan_to_num_nan(self):
        self.check_unary_nan('nan_to_num')

    def test_nan_to_num_inf_nan(self):
        self.check_unary_inf_nan('nan_to_num')
