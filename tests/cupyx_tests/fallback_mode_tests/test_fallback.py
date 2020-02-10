import pytest
import unittest
import functools

import numpy

import cupy
from cupy import testing
from cupyx import fallback_mode
from cupyx.fallback_mode import fallback


def numpy_fallback_equal(name='xp'):
    """
    Decorator that checks fallback_mode results are equal to NumPy ones.
    Checks results that are non-ndarray.

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

            assert numpy_result == fallback_result

        return test_func
    return decorator


def numpy_fallback_array_equal(name='xp'):
    """
    Decorator that checks fallback_mode results are equal to NumPy ones.
    Checks ndarrays.

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
                assert isinstance(fallback_result, fallback.ndarray)
                assert fallback_result.dtype is numpy_result.dtype
                testing.assert_array_equal(
                    numpy_result, fallback_result._array)

            elif isinstance(numpy_result, numpy.ScalarType):
                # if numpy returns scalar
                # cupy may return 0-dim array
                assert numpy_result == fallback_result._array.item()

            else:
                assert False

        return test_func
    return decorator


def numpy_fallback_array_allclose(name='xp'):
    """
    Decorator that checks fallback_mode results are almost equal to NumPy ones.
    Checks ndarrays.

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

            assert isinstance(fallback_result, fallback.ndarray)
            assert fallback_result.dtype is numpy_result.dtype
            testing.numpy_cupy_allclose(
                numpy_result, fallback_result._array)

        return test_func
    return decorator


@testing.gpu
class TestFallbackMode(unittest.TestCase):

    def test_module_not_callable(self):

        pytest.raises(TypeError, fallback_mode.numpy)

        pytest.raises(TypeError, fallback_mode.numpy.linalg)

    def test_numpy_scalars(self):

        assert fallback_mode.numpy.inf is numpy.inf

        assert fallback_mode.numpy.pi is numpy.pi

        # True, because 'is' checks for reference
        # fallback_mode.numpy.nan and numpy.nan both have same reference
        assert fallback_mode.numpy.nan is numpy.nan

    def test_cupy_specific_func(self):

        with pytest.raises(AttributeError):
            func = fallback_mode.numpy.ElementwiseKernel  # NOQA

    def test_func_not_in_numpy(self):

        with pytest.raises(AttributeError):
            func = fallback_mode.numpy.dummy  # NOQA

    def test_same_reference(self):

        assert fallback_mode.numpy.int64 is numpy.int64

        assert fallback_mode.numpy.float32 is numpy.float32

    def test_isinstance(self):

        a = fallback_mode.numpy.float64(3)
        assert isinstance(a, fallback_mode.numpy.float64)

        abs = fallback_mode.numpy.vectorize(fallback_mode.numpy.abs)
        assert isinstance(abs, fallback_mode.numpy.vectorize)

        date = fallback_mode.numpy.datetime64('2019-07-18')
        assert isinstance(date, fallback_mode.numpy.datetime64)


@testing.parameterize(
    {'func': 'min', 'shape': (3, 4), 'args': (), 'kwargs': {'axis': 0}},
    {'func': 'argmin', 'shape': (3, 4), 'args': (), 'kwargs': {}},
    {'func': 'roots', 'shape': (3,), 'args': (), 'kwargs': {}},
    {'func': 'resize', 'shape': (2, 6), 'args': ((6, 2),), 'kwargs': {}},
    {'func': 'resize', 'shape': (3, 4), 'args': ((4, 9),), 'kwargs': {}},
    {'func': 'delete', 'shape': (5, 4), 'args': (1, 0), 'kwargs': {}},
    {'func': 'append', 'shape': (2, 3), 'args': ([[7, 8, 9]],),
     'kwargs': {'axis': 0}},
    {'func': 'asarray_chkfinite', 'shape': (2, 4), 'args': (),
     'kwargs': {'dtype': numpy.float64}}
)
@testing.gpu
class TestFallbackMethodsArrayExternal(unittest.TestCase):

    @numpy_fallback_array_equal()
    def test_fallback_methods_array_external(self, xp):

        a = testing.shaped_random(self.shape, xp=xp, dtype=numpy.int64)

        return getattr(xp, self.func)(a, *self.args, **self.kwargs)


@testing.parameterize(
    {'object': fallback_mode.numpy.ndarray},
    {'object': fallback_mode.numpy.ndarray.__add__},
    {'object': fallback_mode.numpy.vectorize},
    {'object': fallback_mode.numpy.linalg.eig},
)
@testing.gpu
class TestDocs(unittest.TestCase):

    @numpy_fallback_equal()
    def test_docs(self, xp):
        return getattr(self.object, '__doc__')


@testing.gpu
class FallbackArray(unittest.TestCase):

    def test_ndarray_creation(self):

        a = fallback_mode.numpy.array([[1, 2], [3, 4]])
        assert isinstance(a, fallback.ndarray)

        b = fallback_mode.numpy.arange(9)
        assert isinstance(b, fallback.ndarray)
        assert isinstance(b._array, cupy.ndarray)

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

    @numpy_fallback_array_allclose()
    def test_ndarray_init(self, xp):
        return xp.ndarray((3, 4))

    @numpy_fallback_equal()
    def test_ndarray_shape_creation(self, xp):
        a = xp.ndarray((4, 5))
        return a.shape

    def test_instancecheck_ndarray(self):

        a = fallback_mode.numpy.array([1, 2, 3])
        assert isinstance(a, fallback_mode.numpy.ndarray)

        b = fallback_mode.numpy.ndarray((2, 3))
        assert isinstance(b, fallback_mode.numpy.ndarray)

    def test_instancecheck_type(self):
        a = fallback_mode.numpy.arange(3)
        assert isinstance(a, type(a))

    @numpy_fallback_array_allclose()
    def test_type_call(self, xp):
        a = xp.array([1])
        return type(a)((2, 3))

    @numpy_fallback_equal
    def test_type_assert(self, xp):
        a = xp.array([1, 2, 3])
        return type(a) == xp.ndarray


@testing.parameterize(
    {'func': 'min', 'shape': (5,), 'args': (), 'kwargs': {}},
    {'func': 'argmax', 'shape': (5, 3), 'args': (), 'kwargs': {'axis': 0}},
    {'func': 'ptp', 'shape': (3, 3), 'args': (), 'kwargs': {'axis': 1}},
    {'func': 'compress', 'shape': (3, 2), 'args': ([False, True]),
     'kwargs': {'axis': 0}}
)
@testing.gpu
class TestFallbackArrayMethodsInternal(unittest.TestCase):

    @numpy_fallback_array_equal
    def test_fallback_array_methods_internal(self, xp):

        a = testing.shaped_random(self.shape, xp=xp)

        return getattr(a, self.func)(*self.args, **self.kwargs)


@testing.parameterize(
    {'func': '__eq__', 'shape': (3, 4)},
    {'func': '__ne__', 'shape': (3, 1)},
    {'func': '__gt__', 'shape': (4,)},
    {'func': '__lt__', 'shape': (1, 1)},
    {'func': '__ge__', 'shape': (1, 2)},
    {'func': '__le__', 'shape': (1,)}
)
@testing.gpu
class TestArrayComparison(unittest.TestCase):

    @numpy_fallback_array_equal()
    def test_ndarray_comparison(self, xp):

        a = testing.shaped_random(self.shape, xp=xp)
        b = testing.shaped_random(self.shape, xp=xp, seed=3)

        return getattr(a, self.func)(b)


@testing.parameterize(
    {'func': '__str__', 'shape': (5, 6)},
    {'func': '__repr__', 'shape': (3, 4)},
    {'func': '__int__', 'shape': (1,)},
    {'func': '__float__', 'shape': (1, 1)},
    {'func': '__len__', 'shape': (3, 3)},
    {'func': '__bool__', 'shape': (1,)},
)
@testing.gpu
class TestArrayUnaryMethods(unittest.TestCase):

    @numpy_fallback_equal()
    def test_unary_methods(self, xp):
        a = testing.shaped_random(self.shape, xp=xp)
        return getattr(a, self.func)()


@testing.parameterize(
    {'func': '__abs__', 'shape': (5, 6), 'dtype': numpy.float32},
    {'func': '__copy__', 'shape': (3, 4), 'dtype': numpy.float32},
    {'func': '__neg__', 'shape': (3, 3), 'dtype': numpy.float32},
    {'func': '__invert__', 'shape': (2, 4), 'dtype': numpy.int32}
)
@testing.gpu
class TestArrayUnaryMethodsArray(unittest.TestCase):

    @numpy_fallback_array_equal()
    def test_unary_methods_array(self, xp):

        a = testing.shaped_random(self.shape, xp=xp, dtype=self.dtype)

        return getattr(a, self.func)()


@testing.parameterize(
    {'func': '__add__', 'shape': (3, 4), 'dtype': numpy.float32},
    {'func': '__sub__', 'shape': (2, 2), 'dtype': numpy.float32},
    {'func': '__mul__', 'shape': (5, 6), 'dtype': numpy.float32},
    {'func': '__mod__', 'shape': (3, 4), 'dtype': numpy.float32},
    {'func': '__iadd__', 'shape': (1,), 'dtype': numpy.float32},
    {'func': '__imul__', 'shape': (1, 1), 'dtype': numpy.float32},
    {'func': '__and__', 'shape': (3, 3), 'dtype': numpy.int32},
    {'func': '__ipow__', 'shape': (4, 5), 'dtype': numpy.int32},
    {'func': '__xor__', 'shape': (4, 4), 'dtype': numpy.int32},
    {'func': '__lshift__', 'shape': (2,), 'dtype': numpy.int32},
    {'func': '__irshift__', 'shape': (3, 2), 'dtype': numpy.int32},
)
@testing.gpu
class TestArrayArithmeticMethods(unittest.TestCase):

    @numpy_fallback_array_equal()
    def test_arithmetic_methods(self, xp):
        a = testing.shaped_random(self.shape, xp=xp, dtype=self.dtype)
        b = testing.shaped_random(self.shape, xp=xp, dtype=self.dtype, seed=5)
        return getattr(a, self.func)(b)


@testing.gpu
class TestArrayMatmul(unittest.TestCase):

    @testing.with_requires('numpy>=1.16')
    @numpy_fallback_array_allclose()
    def test_mm_matmul(self, xp):
        a = testing.shaped_random((4, 5), xp)
        b = testing.shaped_random((5, 3), xp, seed=5)

        return a.__matmul__(b)


@testing.gpu
class TestVectorizeWrapper(unittest.TestCase):

    @numpy_fallback_array_equal()
    def test_pyfunc_custom_list(self, xp):

        def function(a, b):
            if a > b:
                return a - b
            return a + b

        return xp.vectorize(function)([1, 2, 3, 4], 2)

    @numpy_fallback_array_equal()
    def test_pyfunc_builtin(self, xp):
        a = testing.shaped_random((4, 5), xp)
        vabs = xp.vectorize(abs)
        return vabs(a)

    @numpy_fallback_array_equal()
    def test_pyfunc_numpy(self, xp):
        a = testing.shaped_random((4, 5), xp)
        vabs = xp.vectorize(numpy.abs)
        return vabs(a)

    @numpy_fallback_equal()
    def test_getattr(self, xp):
        vabs = xp.vectorize(numpy.abs)
        return vabs.pyfunc

    @numpy_fallback_array_equal()
    def test_setattr(self, xp):
        a = xp.array([-1, 2, -3])
        vabs = xp.vectorize(abs)
        vabs.otypes = ['float']
        return vabs(a)

    @numpy_fallback_equal()
    def test_doc(self, xp):
        vabs = xp.vectorize(abs)
        return vabs.__doc__
