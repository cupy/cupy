import unittest
import functools
import contextlib
from io import StringIO

import numpy

import cupy
from cupy import testing
from cupyx import fallback_mode


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
                    assert isinstance(fallback_result, cupy.ndarray)
                    testing.assert_array_equal(numpy_result, fallback_result)

                elif isinstance(numpy_result, numpy.ScalarType):
                    # if numpy returns scalar
                    # cupy must return scalar or 0-dim array
                    if isinstance(fallback_result, numpy.ScalarType):
                        assert numpy_result == fallback_result

                    else:
                        # cupy 0-dim array
                        assert numpy_result == int(fallback_result)
                else:
                    raise NotImplementedError

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
            fallback_mode.numpy.ElementwiseKernel

    def test_func_not_in_numpy(self):

        with self.assertRaises(AttributeError):
            fallback_mode.numpy.dummy


@testing.gpu
class TestNotifications(unittest.TestCase):

    def test_seterr_geterr(self):

        default = fallback_mode.geterr()
        assert default == 'warn'

        old = fallback_mode.seterr('ignore')
        current = fallback_mode.geterr()
        assert old == 'warn'
        assert current == 'ignore'
        fallback_mode.seterr(old)

    def test_geterrcall(self):

        def f(func):
            pass

        fallback_mode.seterrcall(f)
        current = fallback_mode.geterrcall()

        assert current == f

    def test_errstate(self):

        old = fallback_mode.seterr('print')
        before = fallback_mode.geterr()

        with fallback_mode.errstate('log'):
            inside = fallback_mode.geterr()
            assert inside == 'log'

        after = fallback_mode.geterr()
        assert before == after
        fallback_mode.seterr(old)

    def test_errstate_func(self):

        def f(func):
            pass

        old = fallback_mode.seterr('call')
        fallback_mode.seterrcall(f)

        before = fallback_mode.geterr()
        before_func = fallback_mode.geterrcall()

        class L:
            def write(msg):
                pass

        log_obj = L()

        with fallback_mode.errstate('log', log_obj):
            inside = fallback_mode.geterr()
            inside_func = fallback_mode.geterrcall()

            assert inside == 'log'
            assert inside_func is log_obj

        after = fallback_mode.geterr()
        after_func = fallback_mode.geterrcall()

        assert before == after
        assert before_func is after_func
        fallback_mode.seterr(old)


@testing.parameterize(
    {'func': fallback_mode.numpy.array_equal, 'shape': (3, 4)},
    {'func': fallback_mode.numpy.array_equiv, 'shape': (3, 4)},
    {'func': fallback_mode.numpy.cross, 'shape': (2, 3)},
    {'func': fallback_mode.numpy.convolve, 'shape': (5,)}
)
@testing.gpu
class TestNotificationModes(unittest.TestCase):

    def test_notification_ignore(self):

        old = fallback_mode.seterr('ignore')
        saved_stdout = StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        fallback_mode.seterr(old)
        output = saved_stdout.getvalue().strip()
        assert output == ""

    def test_notification_print(self):

        old = fallback_mode.seterr('print')
        saved_stdout = StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        fallback_mode.seterr(old)
        nf = self.func._numpy_object
        output = saved_stdout.getvalue().strip()
        assert output == (f"Warning: '{nf.__name__}' method not in cupy, " +
                          f"falling back to '{nf.__module__}.{nf.__name__}'")

    def test_notification_warn(self):

        fallback_mode.seterr('warn')

        with self.assertWarns(fallback_mode.notification.FallbackWarning):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

    def test_notification_raise(self):

        old = fallback_mode.seterr('raise')

        with self.assertRaises(AttributeError):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        fallback_mode.seterr(old)

    def test_notification_call(self):

        def custom_callback(func):
            print("'{}' fallbacked".format(func.__name__))

        old = fallback_mode.seterr('call')
        fallback_mode.seterrcall(custom_callback)
        saved_stdout = StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        fallback_mode.seterr(old)
        nf = self.func._numpy_object
        output = saved_stdout.getvalue().strip()
        assert output == (f"'{nf.__name__}' fallbacked")

    def test_notification_log(self):

        class Log:
            def write(self, msg):
                print("LOG: {}".format(msg))

        log_obj = Log()
        old = fallback_mode.seterr('log')
        fallback_mode.seterrcall(log_obj)
        saved_stdout = StringIO()

        with contextlib.redirect_stdout(saved_stdout):
            a = testing.shaped_random(self.shape, fallback_mode.numpy)
            b = testing.shaped_random(self.shape, fallback_mode.numpy)
            self.func(a, b)

        fallback_mode.seterr(old)
        nf = self.func._numpy_object
        output = saved_stdout.getvalue().strip()
        assert output == (f"LOG: '{nf.__name__}' method not in cupy, " +
                          f"falling back to '{nf.__module__}.{nf.__name__}'")
