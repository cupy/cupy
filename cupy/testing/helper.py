from __future__ import print_function

import functools
import os
import random

import numpy

import cupy
from cupy import internal
from cupy.testing import array
from cupy.testing import parameterized


def _call_func(self, impl, args, kw):
    try:
        result = impl(self, *args, **kw)
        self.assertIsNotNone(result)
        error = None
    except Exception as e:
        result = None
        error = e

    return result, error


def _make_positive_indices(self, impl, args, kw):
    ks = [k for k, v in kw.items() if v in _unsigned_dtypes]
    for k in ks:
        kw[k] = numpy.int64
    mask = cupy.asnumpy(impl(self, *args, **kw)) >= 0
    return numpy.nonzero(mask)


def _contains_signed_and_unsigned(kw):
    vs = set(kw.values())
    return any(d in vs for d in _unsigned_dtypes) and \
        any(d in vs for d in _float_dtypes + _signed_dtypes)


def _make_decorator(check_func, name, type_check, accept_error):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            kw[name] = cupy
            cupy_result, cupy_error = _call_func(self, impl, args, kw)

            kw[name] = numpy
            numpy_result, numpy_error = _call_func(self, impl, args, kw)

            if cupy_error or numpy_error:
                if accept_error:
                    self.assertIs(type(cupy_error), type(numpy_error))
                    return
                elif cupy_error:
                    raise cupy_error
                elif numpy_error:
                    raise numpy_error

            # Behavior of assigning a negative value to an unsigned integer
            # variable is undefined.
            # nVidia GPUs and Intel CPUs behave differently.
            # To avoid this difference, we need to ignore dimensions whose
            # values are negative.
            if _contains_signed_and_unsigned(kw):
                inds = _make_positive_indices(self, impl, args, kw)
                cupy_result = cupy.asnumpy(cupy_result)[inds]
                numpy_result = cupy.asnumpy(numpy_result)[inds]

            check_func(cupy_result, numpy_result)
            if type_check:
                self.assertEqual(cupy_result.dtype, numpy_result.dtype)
        return test_func
    return decorator


def numpy_cupy_allclose(rtol=1e-7, atol=0, err_msg='', verbose=True,
                        name='xp', type_check=True, accept_error=True):
    def check_func(cupy_result, numpy_result):
        array.assert_allclose(cupy_result, numpy_result,
                              rtol, atol, err_msg, verbose)
    return _make_decorator(check_func, name, type_check, accept_error)


def numpy_cupy_array_almost_equal(decimal=6, err_msg='', verbose=True,
                                  name='xp', type_check=True,
                                  accept_error=True):
    def check_func(x, y):
        array.assert_array_almost_equal(
            x, y, decimal, err_msg, verbose)
    return _make_decorator(check_func, name, type_check, accept_error)


def numpy_cupy_arrays_almost_equal_nulp(nulp=1, name='xp', type_check=True,
                                        accept_error=True):
    def check_func(x, y):
        array.assert_arrays_almost_equal_nulp(x, y, nulp)
    return _make_decorator(check_func, name, type_check, accept_error)


def numpy_cupy_array_max_ulp(maxulp=1, dtype=None, name='xp', type_check=True,
                             accept_error=True):
    def check_func(x, y):
        array.assert_array_max_ulp(x, y, maxulp, dtype)
    return _make_decorator(check_func, name, type_check, accept_error)


def numpy_cupy_array_equal(err_msg='', verbose=True, name='xp',
                           type_check=True, accept_error=True):
    def check_func(x, y):
        array.assert_array_equal(x, y, err_msg, verbose)
    return _make_decorator(check_func, name, type_check, accept_error)


def numpy_cupy_array_list_equal(err_msg='', verbose=True, name='xp'):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            kw[name] = cupy
            x = impl(self, *args, **kw)
            kw[name] = numpy
            y = impl(self, *args, **kw)
            self.assertIsNotNone(x)
            self.assertIsNotNone(y)
            array.assert_array_list_equal(x, y, err_msg, verbose)
        return test_func
    return decorator


def numpy_cupy_array_less(err_msg='', verbose=True, name='xp',
                          type_check=True, accept_error=True):
    def check_func(x, y):
        array.assert_array_less(x, y, err_msg, verbose)
    return _make_decorator(check_func, name, type_check, accept_error)


def numpy_cupy_raises(name='xp'):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            kw[name] = cupy
            try:
                impl(self, *args, **kw)
                cupy_error = None
            except Exception as e:
                cupy_error = e

            kw[name] = numpy
            try:
                impl(self, *args, **kw)
                numpy_error = None
            except Exception as e:
                numpy_error = e

            self.assertIs(type(cupy_error), type(numpy_error))
            self.assertIsNotNone(cupy_error)
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


def _make_all_dtypes(no_float16, no_bool):
    if no_float16:
        if no_bool:
            return _regular_float_dtypes + _int_dtypes
        else:
            return _regular_dtypes
    else:
        if no_bool:
            return _float_dtypes + _int_dtypes
        else:
            return _dtypes


def for_all_dtypes(name='dtype', no_float16=False, no_bool=False):
    return for_dtypes(_make_all_dtypes(no_float16, no_bool), name=name)


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


def for_dtypes_combination(types, names=['dtype'], full=None):
    if full is None:
        full = int(os.environ.get('CUPY_TEST_FULL_COMBINATION', '0')) != 0

    if full:
        combination = parameterized.product({name: types for name in names})
    else:
        ts = []
        for _ in range(len(names)):
            # Make shffuled list of types for each name
            t = list(types)
            random.shuffle(t)
            ts.append(t)

        combination = [dict(zip(names, typs)) for typs in zip(*ts)]

    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            for dtypes in combination:
                kw_copy = kw.copy()
                kw_copy.update(dtypes)

                try:
                    impl(self, *args, **kw_copy)
                except Exception:
                    print(dtypes)
                    raise

        return test_func
    return decorator


def for_all_dtypes_combination(names=['dtyes'],
                               no_float16=False, no_bool=False, full=None):
    types = _make_all_dtypes(no_float16, no_bool)
    return for_dtypes_combination(types, names, full)


def for_signed_dtypes_combination(names=['dtype'], full=None):
    return for_dtypes_combination(_signed_dtypes, names=names, full=full)


def for_unsigned_dtypes_combination(names=['dtype'], full=None):
    return for_dtypes_combination(_unsigned_dtypes, names=names, full=full)


def for_int_dtypes_combination(names=['dtype'], no_bool=False, full=None):
    if no_bool:
        types = _int_dtypes
    else:
        types = _int_bool_dtypes
    return for_dtypes_combination(types, names, full)


def shaped_arange(shape, xp=cupy, dtype=numpy.float32):
    a = numpy.arange(1, internal.prod(shape) + 1, 1)
    if numpy.dtype(dtype).type == numpy.bool_:
        return xp.array((a % 2 == 0).reshape(shape))
    else:
        return xp.array(a.astype(dtype).reshape(shape))


def shaped_reverse_arange(shape, xp=cupy, dtype=numpy.float32):
    size = internal.prod(shape)
    a = numpy.arange(size, 0, -1)
    return xp.array(a.astype(dtype).reshape(shape))


def shaped_random(shape, xp=cupy, dtype=numpy.float32, scale=10, seed=0):
    numpy.random.seed(seed)
    if numpy.dtype(dtype).type == numpy.bool_:
        return xp.asarray(numpy.random.randint(2, size=shape).astype(dtype))
    else:
        return xp.asarray((numpy.random.rand(*shape) * scale).astype(dtype))


class NumpyError(object):
    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.err = numpy.geterr()
        numpy.seterr(**self.kw)

    def __exit__(self, exc_type, exc_value, traceback):
        numpy.seterr(**self.err)
