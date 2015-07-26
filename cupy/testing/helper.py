from __future__ import print_function

import functools

import numpy

import cupy
from cupy.testing import array


def numpy_cupy_allclose(rtol=1e-7, atol=0, err_msg='', verbose=True,
                        name='xpy'):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            kw[name] = cupy
            x = impl(self, *args, **kw)
            kw[name] = numpy
            y = impl(self, *args, **kw)
            array.assert_allclose(x, y, rtol, atol, err_msg, verbose)
        return test_func
    return decorator


def numpy_cupy_array_almost_equal(decimal=6, err_msg='', verbose=True,
                                  name='xpy'):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            kw[name] = cupy
            x = impl(self, *args, **kw)
            kw[name] = numpy
            y = impl(self, *args, **kw)
            array.assert_array_almost_equal(
                x, y, decimal, err_msg, verbose)
        return test_func
    return decorator


def numpy_cupy_arrays_almost_equal_nulp(nulp=1, name='xpy'):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            kw[name] = cupy
            x = impl(self, *args, **kw)
            kw[name] = numpy
            y = impl(self, *args, **kw)
            array.assert_arrays_almost_equal_nulp(x, y, nulp)
        return test_func
    return decorator


def numpy_cupy_array_max_ulp(maxulp=1, dtype=None, name='xpy'):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            kw[name] = cupy
            x = impl(self, *args, **kw)
            kw[name] = numpy
            y = impl(self, *args, **kw)
            array.assert_array_max_ulp(x, y, maxulp, dtype)
        return test_func
    return decorator


def numpy_cupy_array_equal(err_msg='', verbose=True, name='xpy'):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            kw[name] = cupy
            x = impl(self, *args, **kw)
            kw[name] = numpy
            y = impl(self, *args, **kw)
            array.assert_array_equal(x, y, err_msg, verbose)
        return test_func
    return decorator


def numpy_cupy_array_list_equal(err_msg='', verbose=True, name='xpy'):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            kw[name] = cupy
            x = impl(self, *args, **kw)
            kw[name] = numpy
            y = impl(self, *args, **kw)
            array.assert_array_list_equal(x, y, err_msg, verbose)
        return test_func
    return decorator


def numpy_cupy_array_less(err_msg='', verbose=True, name='xpy'):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            kw[name] = cupy
            x = impl(self, *args, **kw)
            kw[name] = numpy
            y = impl(self, *args, **kw)
            array.assert_array_less(x, y, err_msg, verbose)
        return test_func
    return decorator


def for_dtypes(dtypes, name='dtype'):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            for dtype in dtypes:
                try:
                    kw[name] = dtype
                    impl(self, *args, **kw)
                except Exception:
                    print(name, 'is', dtype)
                    raise

        return test_func
    return decorator


_regular_float_dtypes = (numpy.float64, numpy.float32)
_float_dtypes = _regular_float_dtypes + (numpy.float16,)
_signed_dtypes = (numpy.int64, numpy.int32, numpy.int16, numpy.int8)
_unsigned_dtypes = (numpy.uint64, numpy.uint32, numpy.uint16, numpy.uint8)
_int_dtypes = _signed_dtypes + _unsigned_dtypes
_int_bool_dtypes = _int_dtypes + (numpy.bool_,)
_regular_dtypes = _regular_float_dtypes + _int_bool_dtypes
_dtypes = _float_dtypes + _int_bool_dtypes


def for_all_dtypes(name='dtype', no_float16=False, no_bool=False):
    if no_float16:
        if no_bool:
            return for_dtypes(_regular_float_dtypes + _int_dtypes, name=name)
        else:
            return for_dtypes(_regular_dtypes, name=name)
    else:
        if no_bool:
            return for_dtypes(_float_dtypes + _int_dtypes, name=name)
        else:
            return for_dtypes(_dtypes, name=name)


def for_float_dtypes(name='dtype', no_float16=False):
    if no_float16:
        return for_dtypes(_regular_float_dtypes, name=name)
    else:
        return for_dtypes(_float_dtypes, name=name)


def for_signed_dtypes(name='dtype'):
    return for_dtypes(_signed_dtypes, name=name)


def for_unsigned_dtypes(name='dtype'):
    return for_dtypes(_unsigned_dtypes, name=name)


def for_int_dtypes(name='dtype', no_bool=False):
    if no_bool:
        return for_dtypes(_int_dtypes, name=name)
    else:
        return for_dtypes(_int_bool_dtypes, name=name)


def shaped_arange(shape, xpy=cupy, dtype=numpy.float32):
    a = numpy.arange(1, numpy.prod(shape, dtype=numpy.int32) + 1, 1)
    if numpy.dtype(dtype).type == numpy.bool_:
        return xpy.array((a % 2 == 0).reshape(shape))
    else:
        return xpy.array(a.astype(dtype).reshape(shape))


def shaped_reverse_arange(shape, xpy=cupy, dtype=numpy.float32):
    size = numpy.prod(shape, dtype=numpy.int32)
    a = numpy.arange(size, 0, -1)
    return xpy.array(a.astype(dtype).reshape(shape))
