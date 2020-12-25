import operator
import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestComparison(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_binary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a, b)

    def test_greater(self):
        self.check_binary('greater')

    def test_greater_equal(self):
        self.check_binary('greater_equal')

    def test_less(self):
        self.check_binary('less')

    def test_less_equal(self):
        self.check_binary('less_equal')

    def test_not_equal(self):
        self.check_binary('not_equal')

    def test_equal(self):
        self.check_binary('equal')


@testing.gpu
class TestComparisonOperator(unittest.TestCase):

    operators = [
        operator.lt, operator.le,
        operator.eq, operator.ne,
        operator.gt, operator.ge,
    ]

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_binary_npscalar_array(self, xp, dtype):
        a = numpy.int16(3)
        b = testing.shaped_arange((2, 3), xp, dtype)
        return [op(a, b) for op in self.operators]

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_binary_pyscalar_array(self, xp, dtype):
        a = 3.0
        b = testing.shaped_arange((2, 3), xp, dtype)
        return [op(a, b) for op in self.operators]

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_binary_array_npscalar(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = numpy.float32(3.0)
        return [op(a, b) for op in self.operators]

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_binary_array_pyscalar(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = 3
        return [op(a, b) for op in self.operators]


class TestArrayEqual(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_array_equal_not_equal(self, xp, dtype):
        a = xp.array([1, 2, 3, 4], dtype=dtype)
        b = xp.array([1, 2, 4, 5], dtype=dtype)
        return xp.array_equal(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_array_equal_is_equal(self, xp, dtype):
        a = xp.array([1, 2, 3, 4], dtype=dtype)
        b = xp.array([1, 2, 3, 4], dtype=dtype)
        return xp.array_equal(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_array_equal_diff_length(self, xp, dtype):
        a = xp.array([1, 2, 3, 4], dtype=dtype)
        b = xp.array([1, 2, 3], dtype=dtype)
        return xp.array_equal(a, b)

    @testing.with_requires('numpy>=1.19')
    @testing.for_float_dtypes()
    @testing.numpy_cupy_equal()
    def test_array_equal_infinite_equal_nan(self, xp, dtype):
        nan = float('nan')
        inf = float('inf')
        ninf = float('-inf')
        a = xp.array([0, nan, inf, ninf], dtype=dtype)
        b = xp.array([0, nan, inf, ninf], dtype=dtype)
        return xp.array_equal(a, b, equal_nan=True)

    @testing.with_requires('numpy>=1.19')
    @testing.for_complex_dtypes()
    @testing.numpy_cupy_equal()
    def test_array_equal_complex_equal_nan(self, xp, dtype):
        a = xp.array([1+2j], dtype=dtype)
        b = a.copy()
        b.imag = xp.nan
        a.real = xp.nan
        return xp.array_equal(a, b, equal_nan=True)

    @testing.numpy_cupy_equal()
    def test_array_equal_diff_dtypes_not_equal(self, xp):
        a = xp.array([0.9e-5, 1.1e-5, 100.5, 10.5])
        b = xp.array([0, 0, 1000, 1000])
        return xp.array_equal(a, b)

    @testing.numpy_cupy_equal()
    def test_array_equal_diff_dtypes_is_equal(self, xp):
        a = xp.array([0.0, 1.0, 100.0, 10.0])
        b = xp.array([0, 1, 100, 10])
        return xp.array_equal(a, b)

    @testing.numpy_cupy_equal()
    def test_array_equal_broadcast_not_allowed(self, xp):
        a = xp.array([1, 1, 1, 1])
        b = xp.array([1])
        return xp.array_equal(a, b)


class TestAllclose(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_allclose_finite(self, xp, dtype):
        a = xp.array([0.9e-5, 1.1e-5, 1000 + 1e-4, 1000 - 1e-4], dtype=dtype)
        b = xp.array([0, 0, 1000, 1000], dtype=dtype)
        return xp.allclose(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_allclose_min_int(self, xp, dtype):
        a = xp.array([0], dtype=dtype)
        b = xp.array([numpy.iinfo('i').min], dtype=dtype)
        return xp.allclose(a, b)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_equal()
    def test_allclose_infinite(self, xp, dtype):
        nan = float('nan')
        inf = float('inf')
        ninf = float('-inf')
        a = xp.array([0, nan, nan, 0, inf, ninf], dtype=dtype)
        b = xp.array([0, nan, 0, nan, inf, ninf], dtype=dtype)
        return xp.allclose(a, b)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_equal()
    def test_allclose_infinite_equal_nan(self, xp, dtype):
        nan = float('nan')
        inf = float('inf')
        ninf = float('-inf')
        a = xp.array([0, nan, inf, ninf], dtype=dtype)
        b = xp.array([0, nan, inf, ninf], dtype=dtype)
        return xp.allclose(a, b, equal_nan=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_allclose_array_scalar(self, xp, dtype):
        a = xp.array([0.9e-5, 1.1e-5], dtype=dtype)
        b = xp.dtype(xp.dtype).type(0)
        return xp.allclose(a, b)


class TestIsclose(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_is_close_finite(self, xp, dtype):
        # In numpy<1.10 this test fails when dtype is bool
        a = xp.array([0.9e-5, 1.1e-5, 1000 + 1e-4, 1000 - 1e-4], dtype=dtype)
        b = xp.array([0, 0, 1000, 1000], dtype=dtype)
        return xp.isclose(a, b)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_is_close_min_int(self, xp, dtype):
        # In numpy<1.10 this test fails when dtype is bool
        a = xp.array([0], dtype=dtype)
        b = xp.array([numpy.iinfo('i').min], dtype=dtype)
        return xp.isclose(a, b)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_is_close_infinite(self, xp, dtype):
        nan = float('nan')
        inf = float('inf')
        ninf = float('-inf')
        a = xp.array([0, nan, nan, 0, inf, ninf], dtype=dtype)
        b = xp.array([0, nan, 0, nan, inf, ninf], dtype=dtype)
        return xp.isclose(a, b)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_is_close_infinite_equal_nan(self, xp, dtype):
        nan = float('nan')
        inf = float('inf')
        ninf = float('-inf')
        a = xp.array([0, nan, inf, ninf], dtype=dtype)
        b = xp.array([0, nan, inf, ninf], dtype=dtype)
        return xp.isclose(a, b, equal_nan=True)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_is_close_array_scalar(self, xp, dtype):
        a = xp.array([0.9e-5, 1.1e-5], dtype=dtype)
        b = xp.dtype(xp.dtype).type(0)
        return xp.isclose(a, b)

    @testing.for_all_dtypes(no_complex=True)
    def test_is_close_scalar_scalar(self, dtype):
        # cupy.isclose always returns ndarray
        a = cupy.dtype(cupy.dtype).type(0)
        b = cupy.dtype(cupy.dtype).type(0)
        cond = cupy.isclose(a, b)
        assert cond.shape == ()
        assert bool(cond)
