import pytest
import unittest
import functools

import numpy

import cupy
from cupy import testing
from cupyx import fallback_mode
from cupyx.fallback_mode import fallback
from cupyx.fallback_mode.notification import FallbackWarning


ignore_fallback_warnings = pytest.mark.filterwarnings(
    "ignore", category=FallbackWarning)


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

                fallback_mode.numpy.testing.assert_array_equal(
                    numpy_result, fallback_result)

                assert fallback_result.dtype == numpy_result.dtype

            elif isinstance(numpy_result, numpy.ScalarType):
                # if numpy returns scalar
                # cupy may return 0-dim array
                assert numpy_result == fallback_result._cupy_array.item() or \
                    (numpy_result == fallback_result._numpy_array).all()

            else:
                assert False

        return test_func
    return decorator


def numpy_fallback_array_allclose(name='xp', rtol=1e-07):
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

            fallback_mode.numpy.testing.assert_allclose(
                numpy_result, fallback_result, rtol=rtol)

            assert fallback_result.dtype == numpy_result.dtype

        return test_func
    return decorator


def enable_slice_copy(func):
    """
    Decorator that enables CUPY_EXPERIMENTAL_SLICE_COPY.
    And then restores it to previous state.
    """
    def decorator(*args, **kwargs):
        old = cupy._util.ENABLE_SLICE_COPY
        cupy._util.ENABLE_SLICE_COPY = True
        func(*args, **kwargs)
        cupy._util.ENABLE_SLICE_COPY = old

    return decorator


def get_numpy_version():
    return tuple(map(int, numpy.__version__.split('.')))


@ignore_fallback_warnings
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
@ignore_fallback_warnings
@testing.gpu
class TestFallbackMethodsArrayExternal(unittest.TestCase):

    @numpy_fallback_array_equal()
    def test_fallback_methods_array_external(self, xp):

        a = testing.shaped_random(self.shape, xp=xp, dtype=numpy.int64)
        return getattr(xp, self.func)(a, *self.args, **self.kwargs)


@testing.parameterize(
    {'func': 'min', 'shape': (3, 4), 'args': (), 'kwargs': {'axis': 0},
     'numpy_version': None},
    {'func': 'argmin', 'shape': (3, 4), 'args': (), 'kwargs': {},
     'numpy_version': (1, 10, 0)},
    {'func': 'arccos', 'shape': (2, 3), 'args': (), 'kwargs': {},
     'numpy_version': None},
    {'func': 'fabs', 'shape': (2, 3), 'args': (), 'kwargs': {},
     'numpy_version': None},
    {'func': 'nancumsum', 'shape': (5, 3), 'args': (), 'kwargs': {'axis': 1},
     'numpy_version': (1, 12, 0)},
    {'func': 'nanpercentile', 'shape': (3, 4), 'args': (50,),
     'kwargs': {'axis': 0}, 'numpy_version': None}
)
@ignore_fallback_warnings
@testing.gpu
class TestFallbackMethodsArrayExternalOut(unittest.TestCase):

    @numpy_fallback_array_equal()
    def test_fallback_methods_array_external_out(self, xp):
        if self.numpy_version and get_numpy_version() < self.numpy_version:
            self.skipTest('Test not supported for this version of numpy')

        a = testing.shaped_random(self.shape, xp=xp)
        kwargs = self.kwargs.copy()
        res = getattr(xp, self.func)(a, *self.args, **kwargs)

        # to get the shape of out
        out = xp.zeros(res.shape, dtype=res.dtype)
        kwargs['out'] = out
        getattr(xp, self.func)(a, *self.args, **kwargs)
        return out


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
class TestFallbackArray(unittest.TestCase):

    def test_ndarray_creation_compatible(self):

        a = fallback_mode.numpy.array([[1, 2], [3, 4]])
        assert isinstance(a, fallback.ndarray)
        assert a._supports_cupy

        b = fallback_mode.numpy.arange(9)
        assert isinstance(b, fallback.ndarray)
        assert a._supports_cupy

    def test_ndarray_creation_not_compatible(self):

        a = fallback_mode.numpy.array([1, 2, 3], dtype=object)
        assert isinstance(a, fallback.ndarray)
        assert not a._supports_cupy

        b = fallback_mode.numpy.array(['a', 'b', 'c', 'd'], dtype='|S1')
        assert isinstance(b, fallback.ndarray)
        assert not b._supports_cupy

        # Structured array will automatically be _numpy_array
        c = fallback_mode.numpy.array(
            [('Rex', 9, 81.0), ('Fido', 3, 27.0)],
            dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])

        assert isinstance(c, fallback.ndarray)
        assert not c._supports_cupy

    def test_getitem(self):

        x = fallback_mode.numpy.array([1, 2, 3])

        # single element
        assert int(x[2]) == 3

        # slicing
        res = cupy.array([1, 2, 3])
        testing.assert_array_equal(x[:2]._cupy_array, res[:2])

    def test_setitem(self):

        x = fallback_mode.numpy.array([1, 2, 3])

        # single element
        x[2] = 99
        res = cupy.array([1, 2, 99])
        testing.assert_array_equal(x._cupy_array, res)

        # slicing
        y = fallback_mode.numpy.array([11, 22])
        x[:2] = y
        res = cupy.array([11, 22, 99])
        testing.assert_array_equal(x._cupy_array, res)

    @numpy_fallback_equal()
    def test_ndarray_shape(self, xp):

        x = xp.arange(20)
        x = x.reshape(4, 5)

        return x.shape

    @numpy_fallback_equal()
    def test_ndarray_init(self, xp):
        a = xp.ndarray((3, 4))
        return a.shape

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

    @numpy_fallback_equal()
    def test_type_call(self, xp):
        a = xp.array([1])
        b = type(a)((2, 3))
        return b.shape

    @numpy_fallback_equal()
    def test_type_assert(self, xp):
        a = xp.array([1, 2, 3])
        return type(a) == xp.ndarray

    @numpy_fallback_equal()
    def test_base(self, xp):
        a = xp.arange(7)
        b = a[2:]
        return b.base is a


@testing.parameterize(
    {'func': 'min', 'shape': (5,), 'args': (), 'kwargs': {}},
    {'func': 'argmax', 'shape': (5, 3), 'args': (), 'kwargs': {'axis': 0}},
    {'func': 'ptp', 'shape': (3, 3), 'args': (), 'kwargs': {'axis': 1}},
    {'func': 'compress', 'shape': (3, 2), 'args': ([False, True],),
     'kwargs': {'axis': 0}}
)
@testing.gpu
class TestFallbackArrayMethodsInternal(unittest.TestCase):

    @numpy_fallback_array_equal()
    def test_fallback_array_methods_internal(self, xp):

        a = testing.shaped_random(self.shape, xp=xp)
        return getattr(a, self.func)(*self.args, **self.kwargs)

    @numpy_fallback_array_equal()
    def test_fallback_array_methods_internal_out(self, xp):

        a = testing.shaped_random(self.shape, xp=xp)
        kwargs = self.kwargs.copy()
        res = getattr(a, self.func)(*self.args, **kwargs)

        # to get the shape of out
        out = xp.zeros(res.shape, dtype=res.dtype)
        kwargs['out'] = out
        getattr(a, self.func)(*self.args, **kwargs)
        return out


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
    @numpy_fallback_array_allclose(rtol=1e-05)
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


@ignore_fallback_warnings
@testing.gpu
class TestInplaceSpecialMethods(unittest.TestCase):

    @numpy_fallback_array_equal()
    def test_resize_internal(self, xp):
        a = testing.shaped_random((3, 4), xp)
        a.resize(4, 5, refcheck=False)
        return a

    @numpy_fallback_array_equal()
    def test_ndarray_byteswap(self, xp):
        a = testing.shaped_random((4,), xp, dtype=xp.int16)
        return a.byteswap()

    @unittest.skipIf(get_numpy_version() < (1, 13, 0),
                     'inplace kwarg for byteswap was added in numpy v1.13.0')
    @numpy_fallback_array_equal()
    def test_ndarray_byteswap_inplace(self, xp):
        a = testing.shaped_random((4,), xp, dtype=xp.int16)
        a.byteswap(inplace=True)
        return a

    @numpy_fallback_array_equal()
    def test_putmask(self, xp):
        a = testing.shaped_random((3, 4), xp, dtype=xp.int8)
        xp.putmask(a, a > 2, a**2)
        return a

    @unittest.skipIf(get_numpy_version() < (1, 15, 0),
                     'put_along_axis introduced in numpy v1.15.0')
    @numpy_fallback_array_equal()
    def test_put_along_axis(self, xp):
        a = xp.array([[10, 30, 20], [60, 40, 50]])
        ai = xp.expand_dims(xp.argmax(a, axis=1), axis=1)
        xp.put_along_axis(a, ai, 99, axis=1)
        return a

    @unittest.skipIf(get_numpy_version() < (1, 15, 0),
                     'quantile introduced in numpy v1.15.0')
    @numpy_fallback_array_equal()
    def test_out_is_returned_when_fallbacked(self, xp):
        a = testing.shaped_random((3, 4), xp)
        z = xp.zeros((4, ))
        res = xp.quantile(a, 0.5, axis=0, out=z)
        assert res is z
        return res

    @numpy_fallback_array_allclose()
    def test_out_is_returned_when_not_fallbacked(self, xp):
        a = testing.shaped_random((3, 4), xp, dtype=xp.float64)
        z = xp.zeros((4,))
        res = xp.var(a, axis=0, out=z)
        assert res is z
        return res


@ignore_fallback_warnings
@testing.gpu
class TestArrayVariants(unittest.TestCase):

    @numpy_fallback_array_equal()
    def test_creation_masked(self, xp):
        mx = xp.ma.array([1, 2, 3, 4], mask=[1, 0, 1, 0])
        return mx

    @numpy_fallback_equal()
    def test_method_internal(self, xp):
        mx = xp.ma.array([1, 2, 3, 4], mask=[1, 0, 1, 0])
        return mx.min()

    @numpy_fallback_equal()
    def test_method_internal_not_callable(self, xp):
        mx = xp.ma.array([1, 2, 3, 4], mask=[1, 0, 1, 0])
        return mx.shape

    @numpy_fallback_equal()
    def test_method_external_masked(self, xp):
        mx = xp.ma.array([1, 2, 3, 4], mask=[1, 0, 1, 0])
        return xp.mean(mx)

    @numpy_fallback_array_equal()
    def test_magic_method_masked(self, xp):
        mx = xp.ma.array([1, 2, 3, 4], mask=[1, 0, 1, 0])
        my = xp.ma.array([4, 2, 3, 1], mask=[1, 0, 1, 0])
        return mx >= my

    @numpy_fallback_array_equal()
    def test_creation_char(self, xp):
        cx = xp.char.array(['a', 'b', 'c'], itemsize=3)
        return cx

    @numpy_fallback_array_equal()
    def test_method_external_char(self, xp):
        cx = xp.char.array(['a', 'b', 'c'], itemsize=3)
        cy = xp.char.array(['a', 'b', 'c'], itemsize=3)
        return xp.char.add(cx, cy)

    @numpy_fallback_array_equal()
    def test_magic_method_char(self, xp):
        cx = xp.char.array(['a', 'b', 'c'], itemsize=3)
        cy = xp.char.array(['a', 'b', 'c'], itemsize=3)
        return cx == cy

    @numpy_fallback_array_equal()
    def test_inplace(self, xp):
        x = xp.arange(12).reshape((3, 4))
        mask = xp.zeros_like(x)
        mask[0, :] = 1
        mx = xp.ma.array(x, mask=mask)
        z = xp.ma.zeros((4,))
        xp.nanmean(mx, axis=0, out=z)
        return z

    @numpy_fallback_array_equal()
    def test_matrix_returned(self, xp):
        x = testing.shaped_random((2, 3), xp=xp)
        y = xp.asmatrix(x)

        if xp is fallback_mode.numpy:
            assert x._supports_cupy
            assert isinstance(y, fallback.ndarray)
            assert not y._supports_cupy
            assert y._numpy_array.__class__ is numpy.matrix

        return y

    @numpy_fallback_array_equal()
    def test_record_array(self, xp):
        ra = xp.rec.array([1, 2, 3])
        return ra

    # changes in MaskedArray should be reflected in base ndarray
    @numpy_fallback_array_equal()
    def test_ma_func(self, xp):
        x = xp.array([1, 2, 3, 4])
        x += x
        mx = xp.ma.array(x, mask=[1, 0, 1, 0])
        assert mx.base is x
        mx += mx
        return x

    # changes in base ndarray should be reflected in MaskedArray
    @enable_slice_copy
    @numpy_fallback_array_equal()
    def test_ma_func_inverse(self, xp):
        x = xp.array([1, 2, 3, 4])
        mx = xp.ma.array(x, mask=[1, 0, 1, 0])
        assert mx.base is x
        mx += mx
        x += x
        return mx
