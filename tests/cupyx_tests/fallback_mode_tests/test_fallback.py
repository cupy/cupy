import unittest
import functools

import numpy

import cupy
from cupy import testing
from cupyx import fallback_mode
from cupyx.fallback_mode import utils


@testing.gpu
class TestFallbackMode(unittest.TestCase):

    def numpy_fallback_equal(name='xp'):
        """
        Decorator that checks fallback_mode results are equal to NumPy ones.

        Args:
            name(str): Argument name whose value is either
            ``numpy`` or ``cupy`` module.
        """
        def decorator(impl):
            @functools.wraps(impl)
            def test_func(self, *args, **kwargs):

                kwargs[name] = fallback_mode.numpy
                fallback_result = impl(self, *args, **kwargs)

                kwargs[name] = numpy
                numpy_result = impl(self, *args, **kwargs)

                if isinstance(numpy_result, numpy.ndarray):
                    # if numpy returns ndarray, cupy must return ndarray
                    assert isinstance(fallback_result, utils.ndarray)
                    testing.assert_array_equal(
                        numpy_result, fallback_result._array)

                elif isinstance(numpy_result, numpy.ScalarType):
                    # if numpy returns scalar
                    # cupy must return scalar or 0-dim array
                    if isinstance(fallback_result, numpy.ScalarType):
                        assert numpy_result == fallback_result

                    else:
                        # cupy 0-dim array
                        assert numpy_result == int(fallback_result._array)
                else:
                    assert numpy_result == fallback_result

            return test_func
        return decorator

    @numpy_fallback_equal()
    def test_argmin(self, xp):

        a = xp.array([
            [13, 5, 45, 23, 9],
            [-5, 41, 0, 22, 4],
            [2, 6, 43, 11, -1]
        ])

        return xp.argmin(a, axis=1)

    @numpy_fallback_equal()
    def test_argmin_zero_dim_array_vs_scalar(self, xp):

        a = xp.array([
            [13, 5, 45, 23, 9],
            [-5, 41, 0, 22, 4],
            [2, 6, 43, 11, -1]
        ])

        return xp.argmin(a)

    # cupy.argmin raises error if list passed, numpy does not
    def test_argmin_list(self):

        a = [
            [13, 5, 45, 23, 9],
            [-5, 41, 0, 22, 4],
            [2, 6, 43, 11, -1]
        ]

        with self.assertRaises(AttributeError):
            fallback_mode.numpy.argmin(a)

        assert numpy.argmin([1, 0, 3]) == 1

    # Non-existing function
    @numpy_fallback_equal()
    def test_array_equal(self, xp):

        a = xp.array([1, 2])
        b = xp.array([1, 2])

        return xp.array_equal(a, b)

    # Both cupy and numpy return 0-d array
    @numpy_fallback_equal()
    def test_convolve_zero_dim_array(self, xp):

        a = xp.array([1, 2, 3])
        b = xp.array([0, 1, 0.5])

        return xp.convolve(a, b, 'valid')

    def test_vectorize(self):

        def function(a, b):
            if a > b:
                return a - b
            return a + b

        actual = numpy.vectorize(function)([1, 2, 3, 4], 2)
        expected = fallback_mode.numpy.vectorize(function)([1, 2, 3, 4], 2)

        assert isinstance(actual, numpy.ndarray)

        # ([1,2,3,4], 2) are arguments to
        # numpy.vectorize(function), not numpy.vectorize
        # So, it returns numpy.ndarray
        assert isinstance(expected, numpy.ndarray)

        testing.assert_array_equal(expected, actual)

    def test_module_not_callable(self):

        self.assertRaises(TypeError, fallback_mode.numpy)

        self.assertRaises(TypeError, fallback_mode.numpy.linalg)

    def test_numpy_scalars(self):

        assert fallback_mode.numpy.inf is numpy.inf

        assert fallback_mode.numpy.pi is numpy.pi

        # True, because 'is' checks for reference
        # fallback_mode.numpy.nan and numpy.nan both have same reference
        assert fallback_mode.numpy.nan is numpy.nan

    def test_cupy_specific_func(self):

        with self.assertRaises(AttributeError):
            func = fallback_mode.numpy.ElementwiseKernel  # NOQA

    def test_func_not_in_numpy(self):

        with self.assertRaises(AttributeError):
            func = fallback_mode.numpy.dummy  # NOQA

    def test_ndarray_creation(self):

        a = fallback_mode.numpy.array([[1, 2], [3, 4]])
        assert isinstance(a, utils.ndarray)

        b = fallback_mode.numpy.arange(9)
        assert isinstance(b, utils.ndarray)
        assert isinstance(b._array, cupy.ndarray)

    # ndarray fallback method
    @numpy_fallback_equal()
    def test_ndarray_tobytes(self, xp):

        a = xp.array([1, 2, 3])

        return a.tobytes()

    @numpy_fallback_equal()
    def test_ndarray_min(self, xp):

        a = xp.array([1, 2, 0, 4])

        return a.min()

    @numpy_fallback_equal()
    def test_ndarray_argmin(self, xp):

        a = xp.array([[1, 2, 3], [7, 8, 9]])

        return a.argmin()

    @numpy_fallback_equal()
    def test_ndarray_argmin_kwargs(self, xp):

        a = xp.array([[1, 2, 3], [7, 8, 9]])

        return a.argmin(axis=0)

    def test_magic_methods(self):

        a = fallback_mode.numpy.array([1, 2, 3])
        b = fallback_mode.numpy.array([9, 8, 7])

        # __add__
        x = a + b
        res = cupy.array([10, 10, 10])
        assert isinstance(x, utils.ndarray)
        testing.assert_array_equal(x._array, res)

        # __iadd__
        x += x
        res += res
        testing.assert_array_equal(x._array, res)

        # __mul__ with integer
        x = x * 5
        res = res * 5
        testing.assert_array_equal(x._array, res)

        # __str__
        assert str(x) == str(res)

        # __neg__ single arg: self
        testing.assert_array_equal((-x)._array, -res)

        # __len__
        assert len(x) == len(res)

        # conversion __int__, __float__
        assert int(fallback_mode.numpy.array([3])) == 3
        assert float(fallback_mode.numpy.array([3])) == 3.0

    def test_getitem(self):

        x = fallback_mode.numpy.array([1, 2, 3])

        # single element
        assert int(x[2]) == 3

        # slicing
        res = cupy.array([1, 2, 3])
        testing.assert_array_equal(x[:2]._array, res[:2])

    def test_setitem(self):

        x = fallback_mode.numpy.array([1, 2, 3])

        # single element
        x[2] = 99
        res = cupy.array([1, 2, 99])
        testing.assert_array_equal(x._array, res)

        # slicing
        y = fallback_mode.numpy.array([11, 22])
        x[:2] = y
        res = cupy.array([11, 22, 99])
        testing.assert_array_equal(x._array, res)

    @numpy_fallback_equal()
    def test_ndarray_shape(self, xp):

        x = xp.arange(20)
        x = x.reshape(4, 5)

        return x.shape

    def test_ndarray_shape_creation(self):

        a = fallback_mode.numpy.ndarray((4, 5))
        b = cupy.ndarray((4, 5))

        testing.assert_array_almost_equal(a._array, b)
