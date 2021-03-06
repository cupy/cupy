import sys
import unittest

import numpy
import pytest

import cupy
from cupy.cuda import runtime
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
    def test_clip_min_max_none(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp, dtype)
            with pytest.raises(ValueError):
                a.clip(None, None)

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

    @testing.for_all_dtypes(name='dtype_x', no_bool=True, no_complex=True)
    @testing.for_all_dtypes(name='dtype_y', no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_interp(self, xp, dtype_y, dtype_x):
        # interpolate at points on and outside the boundaries
        x = xp.asarray([0, 1, 2, 4, 6, 8, 9, 10], dtype=dtype_x)
        fx = xp.asarray([1, 3, 5, 7, 9], dtype=dtype_x)
        fy = xp.sin(fx).astype(dtype_y)
        return xp.interp(x, fx, fy)

    @testing.for_all_dtypes(name='dtype_x', no_bool=True, no_complex=True)
    @testing.for_all_dtypes(name='dtype_y', no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_interp_period(self, xp, dtype_y, dtype_x):
        # interpolate at points on and outside the boundaries
        x = xp.asarray([0, 1, 2, 4, 6, 8, 9, 10], dtype=dtype_x)
        fx = xp.asarray([1, 3, 5, 7, 9], dtype=dtype_x)
        fy = xp.sin(fx).astype(dtype_y)
        return xp.interp(x, fx, fy, period=5)

    @testing.for_all_dtypes(name='dtype_x', no_bool=True, no_complex=True)
    @testing.for_all_dtypes(name='dtype_y', no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_interp_left_right(self, xp, dtype_y, dtype_x):
        # interpolate at points on and outside the boundaries
        x = xp.asarray([0, 1, 2, 4, 6, 8, 9, 10], dtype=dtype_x)
        fx = xp.asarray([1, 3, 5, 7, 9], dtype=dtype_x)
        fy = xp.sin(fx).astype(dtype_y)
        left = 10
        right = 20
        return xp.interp(x, fx, fy, left, right)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_all_dtypes(name='dtype_x', no_bool=True, no_complex=True)
    @testing.for_dtypes('efdFD', name='dtype_y')
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_interp_nan_fy(self, xp, dtype_y, dtype_x):
        # interpolate at points on and outside the boundaries
        x = xp.asarray([0, 1, 2, 4, 6, 8, 9, 10], dtype=dtype_x)
        fx = xp.asarray([1, 3, 5, 7, 9], dtype=dtype_x)
        fy = xp.sin(fx).astype(dtype_y)
        fy[0] = fy[2] = fy[-1] = numpy.nan
        return xp.interp(x, fx, fy)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_float_dtypes(name='dtype_x')
    @testing.for_dtypes('efdFD', name='dtype_y')
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_interp_nan_fx(self, xp, dtype_y, dtype_x):
        # interpolate at points on and outside the boundaries
        x = xp.asarray([0, 1, 2, 4, 6, 8, 9, 10], dtype=dtype_x)
        fx = xp.asarray([1, 3, 5, 7, 9], dtype=dtype_x)
        fy = xp.sin(fx).astype(dtype_y)
        fx[-1] = numpy.nan  # x and fx must remain sorted (NaNs are the last)
        return xp.interp(x, fx, fy)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_float_dtypes(name='dtype_x')
    @testing.for_dtypes('efdFD', name='dtype_y')
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_interp_nan_x(self, xp, dtype_y, dtype_x):
        # interpolate at points on and outside the boundaries
        x = xp.asarray([0, 1, 2, 4, 6, 8, 9, 10], dtype=dtype_x)
        fx = xp.asarray([1, 3, 5, 7, 9], dtype=dtype_x)
        fy = xp.sin(fx).astype(dtype_y)
        x[-1] = numpy.nan  # x and fx must remain sorted (NaNs are the last)
        return xp.interp(x, fx, fy)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_all_dtypes(name='dtype_x', no_bool=True, no_complex=True)
    @testing.for_dtypes('efdFD', name='dtype_y')
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_interp_inf_fy(self, xp, dtype_y, dtype_x):
        # interpolate at points on and outside the boundaries
        x = xp.asarray([0, 1, 2, 4, 6, 8, 9, 10], dtype=dtype_x)
        fx = xp.asarray([1, 3, 5, 7, 9], dtype=dtype_x)
        fy = xp.sin(fx).astype(dtype_y)
        fy[0] = fy[2] = fy[-1] = numpy.inf
        return xp.interp(x, fx, fy)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_float_dtypes(name='dtype_x')
    @testing.for_dtypes('efdFD', name='dtype_y')
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_interp_inf_fx(self, xp, dtype_y, dtype_x):
        # interpolate at points on and outside the boundaries
        x = xp.asarray([0, 1, 2, 4, 6, 8, 9, 10], dtype=dtype_x)
        fx = xp.asarray([1, 3, 5, 7, 9], dtype=dtype_x)
        fy = xp.sin(fx).astype(dtype_y)
        fx[-1] = numpy.inf  # x and fx must remain sorted
        return xp.interp(x, fx, fy)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_float_dtypes(name='dtype_x')
    @testing.for_dtypes('efdFD', name='dtype_y')
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_interp_inf_x(self, xp, dtype_y, dtype_x):
        # interpolate at points on and outside the boundaries
        x = xp.asarray([0, 1, 2, 4, 6, 8, 9, 10], dtype=dtype_x)
        fx = xp.asarray([1, 3, 5, 7, 9], dtype=dtype_x)
        fy = xp.sin(fx).astype(dtype_y)
        x[-1] = numpy.inf  # x and fx must remain sorted
        return xp.interp(x, fx, fy)

    @testing.for_all_dtypes(name='dtype_x', no_bool=True, no_complex=True)
    @testing.for_all_dtypes(name='dtype_y', no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_interp_size1(self, xp, dtype_y, dtype_x):
        # interpolate at points on and outside the boundaries
        x = xp.asarray([0, 1, 2, 4, 6, 8, 9, 10], dtype=dtype_x)
        fx = xp.asarray([5], dtype=dtype_x)
        fy = xp.sin(fx).astype(dtype_y)
        left = 10
        right = 20
        return xp.interp(x, fx, fy, left, right)

    @testing.with_requires('numpy>=1.17.0')
    @testing.for_float_dtypes(name='dtype_x')
    @testing.for_dtypes('efdFD', name='dtype_y')
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_interp_inf_to_nan(self, xp, dtype_y, dtype_x):
        # from NumPy's test_non_finite_inf
        x = xp.asarray([0.5], dtype=dtype_x)
        fx = xp.asarray([-numpy.inf, numpy.inf], dtype=dtype_x)
        fy = xp.asarray([0, 10], dtype=dtype_y)
        return xp.interp(x, fx, fy)


@testing.gpu
@testing.parameterize(*testing.product({
    'mode': ['valid', 'same', 'full'],
    'shape1': [(), (5,), (6,), (20,), (21,)],
    'shape2': [(), (5,), (6,), (20,), (21,)],
}))
class TestConvolveShapeCombination(unittest.TestCase):

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3)
    def test_convolve(self, xp, dtype):
        a = testing.shaped_arange(self.shape1, xp, dtype)
        b = testing.shaped_arange(self.shape2, xp, dtype)
        return xp.convolve(a, b, mode=self.mode)


@testing.gpu
@testing.parameterize(*testing.product({
    'mode': ['valid', 'same', 'full']
}))
class TestConvolve(unittest.TestCase):

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_convolve_non_contiguous(self, xp, dtype):
        a = testing.shaped_arange((300,), xp, dtype)
        b = testing.shaped_arange((100,), xp, dtype)
        return xp.convolve(a[::200], b[10::70], mode=self.mode)

    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-4)
    def test_convolve_large_non_contiguous(self, xp, dtype):
        a = testing.shaped_arange((10000,), xp, dtype)
        b = testing.shaped_arange((100,), xp, dtype)
        return xp.convolve(a[200::], b[10::70], mode=self.mode)

    @pytest.mark.xfail(runtime.is_hip,
                       reason='HIP/ROCm may have a bug with larger `b`')
    @testing.for_all_dtypes_combination(names=['dtype1', 'dtype2'])
    @testing.numpy_cupy_allclose(rtol=1e-2)
    def test_convolve_diff_types(self, xp, dtype1, dtype2):
        a = testing.shaped_random((200,), xp, dtype1)
        b = testing.shaped_random((100,), xp, dtype2)
        return xp.convolve(a, b, mode=self.mode)


@testing.gpu
@testing.parameterize(*testing.product({
    'mode': ['valid', 'same', 'full']
}))
class TestConvolveInvalid(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_convolve_empty(self, dtype):
        for xp in (numpy, cupy):
            a = xp.zeros((0,), dtype)
            with pytest.raises(ValueError):
                xp.convolve(a, a, mode=self.mode)

    @testing.for_all_dtypes()
    def test_convolve_ndim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp, dtype)
            b = testing.shaped_arange((10, 5), xp, dtype)
            with pytest.raises(ValueError):
                xp.convolve(a, b, mode=self.mode)
