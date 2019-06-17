import unittest

import cupy
from cupy import testing
from cupy.fallback_mode import numpy as fb

import numpy


@testing.gpu
class TestFallbackMode(unittest.TestCase):

    def check_func_returning_array(self, func_name, cupy_args, cupy_kwargs,
                                   numpy_args, numpy_kwargs):
        """
        Check functions which returns ndarrays for equal arrays.
        Raises an AssertionError if two ndarray objects are not equal.
        """

        # getting via fallback_mode
        x = getattr(fb, func_name)(*cupy_args, **cupy_kwargs)

        # getting via native numpy
        y = getattr(numpy, func_name)(*numpy_args, **numpy_kwargs)

        assert isinstance(x, cupy.ndarray)
        assert isinstance(y, numpy.ndarray)

        numpy.testing.assert_array_equal(cupy.asnumpy(x), y)

    def check_func_returning_non_array(self, func_name, cupy_args, cupy_kwargs,
                                       numpy_args, numpy_kwargs):
        """
        Check functions which does not returns ndarrays.
        Raises AssertionError if two returned objects are not equal.
        """

        # getting via fallback_mode
        x = getattr(fb, func_name)(*cupy_args, **cupy_kwargs)

        # getting via native numpy
        y = getattr(numpy, func_name)(*numpy_args, **numpy_kwargs)

        assert not isinstance(x, cupy.ndarray)
        assert not isinstance(y, numpy.ndarray)

        assert x == y

    def check_cupy_module(self, module_name):
        """
        Checks for cupy module using fallback mode.
        Therefore compares with cupy module.
        Raises AssertionError if two modules are not same.
        """

        # getting via fallback_mode
        expected_module = eval('fb.' + module_name + '._cupy_module')

        # getting via native cupy
        actual_module = getattr(cupy, module_name)

        assert expected_module == actual_module

    def check_numpy_module(self, module_name):
        """
        Checks for numpy module using fallback mode.
        Therefore compares with numpy module.
        Raises AssertionError if two modules are not same.
        """

        # getting via fallback_mode
        expected_module = eval('fb.' + module_name + '._numpy_module')

        # getting via native numpy
        actual_module = getattr(numpy, module_name)

        assert expected_module == actual_module

    # cupy.argmin raises error if list passed, numpy does not
    def test_argmin_list(self):

        a = [
            [13, 5, 45, 23, 9],
            [-5, 41, 0, 22, 4],
            [2, 6, 43, 11, -1]
        ]

        self.assertRaises(AttributeError, fb.argmin, a)
        assert numpy.argmin([1, 0, 3]) == 1

    # Non-existing function
    def test_array_equal(self):

        a = [1, 2]
        b = [1, 2]

        cupy_args = (cupy.array(a), cupy.array(b))
        numpy_args = (numpy.array(a), numpy.array(b))

        kwargs = {}

        self.check_func_returning_non_array('array_equal', cupy_args, kwargs,
                                            numpy_args, kwargs)

    # Both cupy and numpy return 0-d array
    def test_convolve_zero_dim_array(self):

        args = ([1, 2, 3], [0, 1, 0.5], 'valid')

        kwargs = {}

        self.check_func_returning_array('convolve', args, kwargs, args, kwargs)

    # Existing function
    """
    This test is going to fail as argmin exist in cupy,
    if used `check_func_returning_array`.
    Because, cupy returns 0-d array and numpy returns scalar value.
    """

    def test_argmin_zero_dim_array_vs_scalar(self):

        a = ([
            [13, 5, 45, 23, 9],
            [-5, 41, 0, 22, 4],
            [2, 6, 43, 11, -1]
        ],)

        cupy_args = (cupy.array(a),)
        numpy_args = (numpy.array(a),)

        kwargs = {}

        # getting via fallback_mode (cupy)
        x = getattr(fb, 'argmin')(*cupy_args, **kwargs)

        # getting via native numpy
        y = getattr(numpy, 'argmin')(*numpy_args, **kwargs)

        assert int(x) == y

    def test_linalg_module(self):
        self.check_cupy_module('linalg')

    def test_random_module(self):
        self.check_numpy_module('random')

    def test_matrixlib_module(self):
        self.check_numpy_module('matrixlib')

    def test_vectorize(self):

        def function(a, b):
            if a > b:
                return a - b
            return a + b

        actual = numpy.vectorize(function)([1, 2, 3, 4], 2)
        expected = fb.vectorize(function)([1, 2, 3, 4], 2)

        assert isinstance(actual, numpy.ndarray)

        # ([1,2,3,4], 2) are arguments to numpy.vectorize(function),
        # not numpy.vectorize
        # returns as numpy.ndarray
        assert isinstance(expected, numpy.ndarray)

        numpy.testing.assert_array_equal(cupy.asnumpy(expected), actual)

    def test_module_not_callable(self):

        self.assertRaises(TypeError, fb)

        self.assertRaises(TypeError, fb.linalg)

        self.assertRaises(TypeError, fb.linalg._cupy_module)

    def test_numpy_scalars(self):

        assert fb.inf is numpy.inf

        assert fb.pi is numpy.pi

        # True, because is checks for reference
        # fb.nan abd numpy.nan both have same reference
        assert fb.nan is numpy.nan

        # But as nan is not comparable
        assert fb.nan != numpy.nan

    def test_cupy_specific_func(self):

        with self.assertRaises(AttributeError):
            func = fb.ElementwiseKernel  # NOQA

    def test_func_not_in_numpy(self):

        with self.assertRaises(AttributeError):
            func = fb.dummy  # NOQA
