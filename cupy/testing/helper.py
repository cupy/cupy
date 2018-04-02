from __future__ import absolute_import
from __future__ import print_function

import contextlib
import functools
import inspect
import os
import pkg_resources
import random
import traceback
import unittest
import warnings

import numpy
import six

import cupy
from cupy import internal
import cupy.sparse
from cupy.testing import array
from cupy.testing import parameterized


def _call_func(self, impl, args, kw):
    try:
        result = impl(self, *args, **kw)
        self.assertIsNotNone(result)
        error = None
        tb = None
    except Exception as e:
        result = None
        error = e
        tb = traceback.format_exc()

    return result, error, tb


def _check_cupy_numpy_error(self, cupy_error, cupy_tb, numpy_error,
                            numpy_tb, accept_error=False):
    # For backward compatibility
    if accept_error is True:
        accept_error = Exception
    elif not accept_error:
        accept_error = ()
    # TODO(oktua): expected_regexp like numpy.testing.assert_raises_regex
    if cupy_error is None and numpy_error is None:
        self.fail('Both cupy and numpy are expected to raise errors, but not')
    elif cupy_error is None:
        self.fail('Only numpy raises error\n\n' + numpy_tb)
    elif numpy_error is None:
        self.fail('Only cupy raises error\n\n' + cupy_tb)
    elif not isinstance(cupy_error, type(numpy_error)):
        # CuPy errors should be at least as explicit as the NumPy errors, i.e.
        # allow CuPy errors to derive from NumPy errors but not the opposite.
        # This ensures that try/except blocks that catch NumPy errors also
        # catch CuPy errors.
        msg = '''Different types of errors occurred

cupy
%s
numpy
%s
''' % (cupy_tb, numpy_tb)
        self.fail(msg)
    elif not (isinstance(cupy_error, accept_error) and
              isinstance(numpy_error, accept_error)):
        msg = '''Both cupy and numpy raise exceptions

cupy
%s
numpy
%s
''' % (cupy_tb, numpy_tb)
        self.fail(msg)


def _make_positive_indices(self, impl, args, kw):
    ks = [k for k, v in kw.items() if v in _unsigned_dtypes]
    for k in ks:
        kw[k] = numpy.intp
    mask = cupy.asnumpy(impl(self, *args, **kw)) >= 0
    return numpy.nonzero(mask)


def _contains_signed_and_unsigned(kw):
    vs = set(kw.values())
    return any(d in vs for d in _unsigned_dtypes) and \
        any(d in vs for d in _float_dtypes + _signed_dtypes)


def _make_decorator(check_func, name, type_check, accept_error, sp_name=None):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            if sp_name:
                kw[sp_name] = cupy.sparse
            kw[name] = cupy
            cupy_result, cupy_error, cupy_tb = _call_func(self, impl, args, kw)

            kw[name] = numpy
            if sp_name:
                import scipy.sparse
                kw[sp_name] = scipy.sparse
            numpy_result, numpy_error, numpy_tb = \
                _call_func(self, impl, args, kw)

            if cupy_error or numpy_error:
                _check_cupy_numpy_error(self, cupy_error, cupy_tb,
                                        numpy_error, numpy_tb,
                                        accept_error=accept_error)
                return

            self.assertEqual(cupy_result.shape, numpy_result.shape)

            # Behavior of assigning a negative value to an unsigned integer
            # variable is undefined.
            # nVidia GPUs and Intel CPUs behave differently.
            # To avoid this difference, we need to ignore dimensions whose
            # values are negative.
            skip = False
            if _contains_signed_and_unsigned(kw) and \
                    cupy_result.dtype in _unsigned_dtypes:
                inds = _make_positive_indices(self, impl, args, kw)
                if cupy_result.shape == ():
                    skip = inds[0].size == 0
                else:
                    cupy_result = cupy.asnumpy(cupy_result)[inds]
                    numpy_result = cupy.asnumpy(numpy_result)[inds]

            if not skip:
                check_func(cupy_result, numpy_result)
            if type_check:
                self.assertEqual(cupy_result.dtype, numpy_result.dtype,
                                 'cupy dtype is not equal to numpy dtype')
        return test_func
    return decorator


def numpy_cupy_allclose(rtol=1e-7, atol=0, err_msg='', verbose=True,
                        name='xp', type_check=True, accept_error=False,
                        sp_name=None, contiguous_check=True):
    """Decorator that checks NumPy results and CuPy ones are close.

    Args:
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupy.sparse`` module. If ``None``, no
             argument is given for the modules.
         contiguous_check(bool): If ``True``, consistency of contiguity is
             also checked.

    Decorated test fixture is required to return the arrays whose values are
    close between ``numpy`` case and ``cupy`` case.
    For example, this test case checks ``numpy.zeros`` and ``cupy.zeros``
    should return same value.

    >>> import unittest
    >>> from cupy import testing
    >>> @testing.gpu
    ... class TestFoo(unittest.TestCase):
    ...
    ...     @testing.numpy_cupy_allclose()
    ...     def test_foo(self, xp):
    ...         # ...
    ...         # Prepare data with xp
    ...         # ...
    ...
    ...         xp_result = xp.zeros(10)
    ...         return xp_result

    .. seealso:: :func:`cupy.testing.assert_allclose`
    """
    def check_func(cupy_result, numpy_result):
        c = cupy_result
        n = numpy_result
        array.assert_allclose(c, n, rtol, atol, err_msg, verbose)
        if contiguous_check and isinstance(n, numpy.ndarray):
            if n.flags.c_contiguous and not c.flags.c_contiguous:
                raise AssertionError(
                    'The state of c_contiguous flag is false. '
                    '(cupy_result:{} numpy_result:{})'.format(
                        c.flags.c_contiguous, n.flags.c_contiguous))
            if n.flags.f_contiguous and not c.flags.f_contiguous:
                raise AssertionError(
                    'The state of f_contiguous flag is false. '
                    '(cupy_result:{} numpy_result:{})'.format(
                        c.flags.f_contiguous, n.flags.f_contiguous))
    return _make_decorator(check_func, name, type_check, accept_error, sp_name)


def numpy_cupy_array_almost_equal(decimal=6, err_msg='', verbose=True,
                                  name='xp', type_check=True,
                                  accept_error=False, sp_name=None):
    """Decorator that checks NumPy results and CuPy ones are almost equal.

    Args:
         decimal(int): Desired precision.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupy.sparse`` module. If ``None``, no
             argument is given for the modules.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`cupy.testing.assert_array_almost_equal`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.

    .. seealso:: :func:`cupy.testing.assert_array_almost_equal`
    """
    def check_func(x, y):
        array.assert_array_almost_equal(
            x, y, decimal, err_msg, verbose)
    return _make_decorator(check_func, name, type_check, accept_error, sp_name)


def numpy_cupy_array_almost_equal_nulp(nulp=1, name='xp', type_check=True,
                                       accept_error=False, sp_name=None):
    """Decorator that checks results of NumPy and CuPy are equal w.r.t. spacing.

    Args:
         nulp(int): The maximum number of unit in the last place for tolerance.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True``, all error types are acceptable.
             If it is ``False``, no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupy.sparse`` module. If ``None``, no
             argument is given for the modules.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`cupy.testing.assert_array_almost_equal_nulp`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.

    .. seealso:: :func:`cupy.testing.assert_array_almost_equal_nulp`
    """
    def check_func(x, y):
        array.assert_array_almost_equal_nulp(x, y, nulp)
    return _make_decorator(check_func, name, type_check, accept_error, sp_name)


def numpy_cupy_array_max_ulp(maxulp=1, dtype=None, name='xp', type_check=True,
                             accept_error=False, sp_name=None):
    """Decorator that checks results of NumPy and CuPy ones are equal w.r.t. ulp.

    Args:
         maxulp(int): The maximum number of units in the last place
             that elements of resulting two arrays can differ.
         dtype(numpy.dtype): Data-type to convert the resulting
             two array to if given.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupy.sparse`` module. If ``None``, no
             argument is given for the modules.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`assert_array_max_ulp`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.

    .. seealso:: :func:`cupy.testing.assert_array_max_ulp`

    """
    def check_func(x, y):
        array.assert_array_max_ulp(x, y, maxulp, dtype)
    return _make_decorator(check_func, name, type_check, accept_error, sp_name)


def numpy_cupy_array_equal(err_msg='', verbose=True, name='xp',
                           type_check=True, accept_error=False, sp_name=None):
    """Decorator that checks NumPy results and CuPy ones are equal.

    Args:
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupy.sparse`` module. If ``None``, no
             argument is given for the modules.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`numpy_cupy_array_equal`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.

    .. seealso:: :func:`cupy.testing.assert_array_equal`
    """
    def check_func(x, y):
        array.assert_array_equal(x, y, err_msg, verbose)

    return _make_decorator(check_func, name, type_check, accept_error, sp_name)


def numpy_cupy_array_list_equal(
        err_msg='', verbose=True, name='xp', sp_name=None):
    """Decorator that checks the resulting lists of NumPy and CuPy's one are equal.

    Args:
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are appended
             to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupy.sparse`` module. If ``None``, no
             argument is given for the modules.

    Decorated test fixture is required to return the same list of arrays
    (except the type of array module) even if ``xp`` is ``numpy`` or ``cupy``.

    .. seealso:: :func:`cupy.testing.assert_array_list_equal`
    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            if sp_name:
                kw[sp_name] = cupy.sparse
            kw[name] = cupy
            x = impl(self, *args, **kw)

            if sp_name:
                import scipy.sparse
                kw[sp_name] = scipy.sparse
            kw[name] = numpy
            y = impl(self, *args, **kw)
            self.assertIsNotNone(x)
            self.assertIsNotNone(y)
            array.assert_array_list_equal(x, y, err_msg, verbose)
        return test_func
    return decorator


def numpy_cupy_array_less(err_msg='', verbose=True, name='xp',
                          type_check=True, accept_error=False, sp_name=None):
    """Decorator that checks the CuPy result is less than NumPy result.

    Args:
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupy.sparse`` module. If ``None``, no
             argument is given for the modules.

    Decorated test fixture is required to return the smaller array
    when ``xp`` is ``cupy`` than the one when ``xp`` is ``numpy``.

    .. seealso:: :func:`cupy.testing.assert_array_less`
    """
    def check_func(x, y):
        array.assert_array_less(x, y, err_msg, verbose)
    return _make_decorator(check_func, name, type_check, accept_error, sp_name)


def numpy_cupy_equal(name='xp', sp_name=None):
    """Decorator that checks NumPy results are equal to CuPy ones.

    Args:
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupy.sparse`` module. If ``None``, no
             argument is given for the modules.

    Decorated test fixture is required to return the same results
    even if ``xp`` is ``numpy`` or ``cupy``.
    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            if sp_name:
                kw[sp_name] = cupy.sparse
            kw[name] = cupy
            cupy_result = impl(self, *args, **kw)

            if sp_name:
                import scipy.sparse
                kw[sp_name] = scipy.sparse
            kw[name] = numpy
            numpy_result = impl(self, *args, **kw)

            if cupy_result != numpy_result:
                message = '''Results are not equal:
cupy: %s
numpy: %s''' % (str(cupy_result), str(numpy_result))
                raise AssertionError(message)
        return test_func
    return decorator


def numpy_cupy_raises(name='xp', sp_name=None, accept_error=Exception):
    """Decorator that checks the NumPy and CuPy throw same errors.

    Args:
         name(str): Argument name whose value is either
             ``numpy`` or ``cupy`` module.
         sp_name(str or None): Argument name whose value is either
             ``scipy.sparse`` or ``cupy.sparse`` module. If ``None``, no
             argument is given for the modules.
         accept_error(bool, Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and CuPy test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.
             If it is ``True`` all error types are acceptable.
             If it is ``False`` no error is acceptable.

    Decorated test fixture is required throw same errors
    even if ``xp`` is ``numpy`` or ``cupy``.
    """

    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            if sp_name:
                kw[sp_name] = cupy.sparse
            kw[name] = cupy
            try:
                impl(self, *args, **kw)
                cupy_error = None
                cupy_tb = None
            except Exception as e:
                cupy_error = e
                cupy_tb = traceback.format_exc()

            if sp_name:
                import scipy.sparse
                kw[sp_name] = scipy.sparse
            kw[name] = numpy
            try:
                impl(self, *args, **kw)
                numpy_error = None
                numpy_tb = None
            except Exception as e:
                numpy_error = e
                numpy_tb = traceback.format_exc()

            _check_cupy_numpy_error(self, cupy_error, cupy_tb,
                                    numpy_error, numpy_tb,
                                    accept_error=accept_error)
        return test_func
    return decorator


def for_dtypes(dtypes, name='dtype'):
    """Decorator for parameterized dtype test.

    Args:
         dtypes(list of dtypes): dtypes to be tested.
         name(str): Argument name to which specified dtypes are passed.

    This decorator adds a keyword argument specified by ``name``
    to the test fixture. Then, it runs the fixtures in parallel
    by passing the each element of ``dtypes`` to the named
    argument.
    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            for dtype in dtypes:
                try:
                    kw[name] = numpy.dtype(dtype).type
                    impl(self, *args, **kw)
                except Exception:
                    print(name, 'is', dtype)
                    raise

        return test_func
    return decorator


_complex_dtypes = (numpy.complex64, numpy.complex128)
_regular_float_dtypes = (numpy.float64, numpy.float32)
_float_dtypes = _regular_float_dtypes + (numpy.float16,)
_signed_dtypes = tuple(numpy.dtype(i).type for i in 'bhilq')
_unsigned_dtypes = tuple(numpy.dtype(i).type for i in 'BHILQ')
_int_dtypes = _signed_dtypes + _unsigned_dtypes
_int_bool_dtypes = _int_dtypes + (numpy.bool_,)
_regular_dtypes = _regular_float_dtypes + _int_bool_dtypes
_dtypes = _float_dtypes + _int_bool_dtypes


def _make_all_dtypes(no_float16, no_bool, no_complex):
    if no_float16:
        dtypes = _regular_float_dtypes
    else:
        dtypes = _float_dtypes

    if no_bool:
        dtypes += _int_dtypes
    else:
        dtypes += _int_bool_dtypes

    if not no_complex:
        dtypes += _complex_dtypes

    return dtypes


def for_all_dtypes(name='dtype', no_float16=False, no_bool=False,
                   no_complex=False):
    """Decorator that checks the fixture with all dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.
         no_float16(bool): If ``True``, ``numpy.float16`` is
             omitted from candidate dtypes.
         no_bool(bool): If ``True``, ``numpy.bool_`` is
             omitted from candidate dtypes.
         no_complex(bool): If ``True``, ``numpy.complex64`` and
             ``numpy.complex128`` are omitted from candidate dtypes.

    dtypes to be tested: ``numpy.complex64`` (optional),
    ``numpy.complex128`` (optional),
    ``numpy.float16`` (optional), ``numpy.float32``,
    ``numpy.float64``, ``numpy.dtype('b')``, ``numpy.dtype('h')``,
    ``numpy.dtype('i')``, ``numpy.dtype('l')``, ``numpy.dtype('q')``,
    ``numpy.dtype('B')``, ``numpy.dtype('H')``, ``numpy.dtype('I')``,
    ``numpy.dtype('L')``, ``numpy.dtype('Q')``, and ``numpy.bool_`` (optional).

    The usage is as follows.
    This test fixture checks if ``cPickle`` successfully reconstructs
    :class:`cupy.ndarray` for various dtypes.
    ``dtype`` is an argument inserted by the decorator.

    >>> import unittest
    >>> from cupy import testing
    >>> @testing.gpu
    ... class TestNpz(unittest.TestCase):
    ...
    ...     @testing.for_all_dtypes()
    ...     def test_pickle(self, dtype):
    ...         a = testing.shaped_arange((2, 3, 4), dtype=dtype)
    ...         s = six.moves.cPickle.dumps(a)
    ...         b = six.moves.cPickle.loads(s)
    ...         testing.assert_array_equal(a, b)

    Typically, we use this decorator in combination with
    decorators that check consistency between NumPy and CuPy like
    :func:`cupy.testing.numpy_cupy_allclose`.
    The following is such an example.

    >>> import unittest
    >>> from cupy import testing
    >>> @testing.gpu
    ... class TestMean(unittest.TestCase):
    ...
    ...     @testing.for_all_dtypes()
    ...     @testing.numpy_cupy_allclose()
    ...     def test_mean_all(self, xp, dtype):
    ...         a = testing.shaped_arange((2, 3), xp, dtype)
    ...         return a.mean()

    .. seealso:: :func:`cupy.testing.for_dtypes`
    """
    return for_dtypes(_make_all_dtypes(no_float16, no_bool, no_complex),
                      name=name)


def for_float_dtypes(name='dtype', no_float16=False):
    """Decorator that checks the fixture with float dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.
         no_float16(bool): If ``True``, ``numpy.float16`` is
             omitted from candidate dtypes.

    dtypes to be tested are ``numpy.float16`` (optional), ``numpy.float32``,
    and ``numpy.float64``.

    .. seealso:: :func:`cupy.testing.for_dtypes`,
        :func:`cupy.testing.for_all_dtypes`
    """
    if no_float16:
        return for_dtypes(_regular_float_dtypes, name=name)
    else:
        return for_dtypes(_float_dtypes, name=name)


def for_signed_dtypes(name='dtype'):
    """Decorator that checks the fixture with signed dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.

    dtypes to be tested are ``numpy.dtype('b')``, ``numpy.dtype('h')``,
    ``numpy.dtype('i')``, ``numpy.dtype('l')``, and ``numpy.dtype('q')``.

    .. seealso:: :func:`cupy.testing.for_dtypes`,
        :func:`cupy.testing.for_all_dtypes`
    """
    return for_dtypes(_signed_dtypes, name=name)


def for_unsigned_dtypes(name='dtype'):
    """Decorator that checks the fixture with unsinged dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.

    dtypes to be tested are ``numpy.dtype('B')``, ``numpy.dtype('H')``,

     ``numpy.dtype('I')``, ``numpy.dtype('L')``, and ``numpy.dtype('Q')``.

    .. seealso:: :func:`cupy.testing.for_dtypes`,
        :func:`cupy.testing.for_all_dtypes`
    """
    return for_dtypes(_unsigned_dtypes, name=name)


def for_int_dtypes(name='dtype', no_bool=False):
    """Decorator that checks the fixture with integer and optionally bool dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.
         no_bool(bool): If ``True``, ``numpy.bool_`` is
             omitted from candidate dtypes.

    dtypes to be tested are ``numpy.dtype('b')``, ``numpy.dtype('h')``,
    ``numpy.dtype('i')``, ``numpy.dtype('l')``, ``numpy.dtype('q')``,
    ``numpy.dtype('B')``, ``numpy.dtype('H')``, ``numpy.dtype('I')``,
    ``numpy.dtype('L')``, ``numpy.dtype('Q')``, and ``numpy.bool_`` (optional).

    .. seealso:: :func:`cupy.testing.for_dtypes`,
        :func:`cupy.testing.for_all_dtypes`
    """
    if no_bool:
        return for_dtypes(_int_dtypes, name=name)
    else:
        return for_dtypes(_int_bool_dtypes, name=name)


def for_complex_dtypes(name='dtype'):
    """Decorator that checks the fixture with complex dtypes.

    Args:
         name(str): Argument name to which specified dtypes are passed.

    dtypes to be tested are ``numpy.complex64`` and ``numpy.complex128``.

    .. seealso:: :func:`cupy.testing.for_dtypes`,
        :func:`cupy.testing.for_all_dtypes`
    """
    return for_dtypes(_complex_dtypes, name=name)


def for_dtypes_combination(types, names=('dtype',), full=None):
    """Decorator that checks the fixture with a product set of dtypes.

    Args:
         types(list of dtypes): dtypes to be tested.
         names(list of str): Argument names to which dtypes are passed.
         full(bool): If ``True``, then all combinations
             of dtypes will be tested.
             Otherwise, the subset of combinations will be tested
             (see the description below).

    Decorator adds the keyword arguments specified by ``names``
    to the test fixture. Then, it runs the fixtures in parallel
    with passing (possibly a subset of) the product set of dtypes.
    The range of dtypes is specified by ``types``.

    The combination of dtypes to be tested changes depending
    on the option ``full``. If ``full`` is ``True``,
    all combinations of ``types`` are tested.
    Sometimes, such an exhaustive test can be costly.
    So, if ``full`` is ``False``, only the subset of possible
    combinations is tested. Specifically, at first,
    the shuffled lists of ``types`` are made for each argument
    name in ``names``.
    Let the lists be ``D1``, ``D2``, ..., ``Dn``
    where :math:`n` is the number of arguments.
    Then, the combinations to be tested will be ``zip(D1, ..., Dn)``.
    If ``full`` is ``None``, the behavior is switched
    by setting the environment variable ``CUPY_TEST_FULL_COMBINATION=1``.

    For example, let ``types`` be ``[float16, float32, float64]``
    and ``names`` be ``['a_type', 'b_type']``. If ``full`` is ``True``,
    then the decorated test fixture is executed with all
    :math:`2^3` patterns. On the other hand, if ``full`` is ``False``,
    shuffled lists are made for ``a_type`` and ``b_type``.
    Suppose the lists are ``(16, 64, 32)`` for ``a_type`` and
    ``(32, 64, 16)`` for ``b_type`` (prefixes are removed for short).
    Then the combinations of ``(a_type, b_type)`` to be tested are
    ``(16, 32)``, ``(64, 64)`` and ``(32, 16)``.
    """

    if full is None:
        full = int(os.environ.get('CUPY_TEST_FULL_COMBINATION', '0')) != 0

    if full:
        combination = parameterized.product({name: types for name in names})
    else:
        ts = []
        for _ in range(len(names)):
            # Make shuffled list of types for each name
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


def for_all_dtypes_combination(names=('dtyes',),
                               no_float16=False, no_bool=False, full=None,
                               no_complex=False):
    """Decorator that checks the fixture with a product set of all dtypes.

    Args:
         names(list of str): Argument names to which dtypes are passed.
         no_float16(bool): If ``True``, ``numpy.float16`` is
             omitted from candidate dtypes.
         no_bool(bool): If ``True``, ``numpy.bool_`` is
             omitted from candidate dtypes.
         full(bool): If ``True``, then all combinations of dtypes
             will be tested.
             Otherwise, the subset of combinations will be tested
             (see description in :func:`cupy.testing.for_dtypes_combination`).
         no_complex(bool): If, True, ``numpy.complex64`` and
             ``numpy.complex128`` are omitted from candidate dtypes.

    .. seealso:: :func:`cupy.testing.for_dtypes_combination`
    """
    types = _make_all_dtypes(no_float16, no_bool, no_complex)
    return for_dtypes_combination(types, names, full)


def for_signed_dtypes_combination(names=('dtype',), full=None):
    """Decorator for parameterized test w.r.t. the product set of signed dtypes.

    Args:
         names(list of str): Argument names to which dtypes are passed.
         full(bool): If ``True``, then all combinations of dtypes
             will be tested.
             Otherwise, the subset of combinations will be tested
             (see description in :func:`cupy.testing.for_dtypes_combination`).

    .. seealso:: :func:`cupy.testing.for_dtypes_combination`
    """
    return for_dtypes_combination(_signed_dtypes, names=names, full=full)


def for_unsigned_dtypes_combination(names=('dtype',), full=None):
    """Decorator for parameterized test w.r.t. the product set of unsigned dtypes.

    Args:
         names(list of str): Argument names to which dtypes are passed.
         full(bool): If ``True``, then all combinations of dtypes
             will be tested.
             Otherwise, the subset of combinations will be tested
             (see description in :func:`cupy.testing.for_dtypes_combination`).

    .. seealso:: :func:`cupy.testing.for_dtypes_combination`
    """
    return for_dtypes_combination(_unsigned_dtypes, names=names, full=full)


def for_int_dtypes_combination(names=('dtype',), no_bool=False, full=None):
    """Decorator for parameterized test w.r.t. the product set of int and boolean.

    Args:
         names(list of str): Argument names to which dtypes are passed.
         no_bool(bool): If ``True``, ``numpy.bool_`` is
             omitted from candidate dtypes.
         full(bool): If ``True``, then all combinations of dtypes
             will be tested.
             Otherwise, the subset of combinations will be tested
             (see description in :func:`cupy.testing.for_dtypes_combination`).

    .. seealso:: :func:`cupy.testing.for_dtypes_combination`
    """
    if no_bool:
        types = _int_dtypes
    else:
        types = _int_bool_dtypes
    return for_dtypes_combination(types, names, full)


def for_orders(orders, name='order'):
    """Decorator to parameterize tests with order.

    Args:
         orders(list of order): orders to be tested.
         name(str): Argument name to which the specified order is passed.

    This decorator adds a keyword argument specified by ``name``
    to the test fixtures. Then, the fixtures run by passing each element of
    ``orders`` to the named argument.

    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(self, *args, **kw):
            for order in orders:
                try:
                    kw[name] = order
                    impl(self, *args, **kw)
                except Exception:
                    print(name, 'is', order)
                    raise

        return test_func
    return decorator


def for_CF_orders(name='order'):
    """Decorator that checks the fixture with orders 'C' and 'F'.

    Args:
         name(str): Argument name to which the specified order is passed.

    .. seealso:: :func:`cupy.testing.for_all_dtypes`

    """
    return for_orders(['C', 'F'], name)


def with_requires(*requirements):
    """Run a test case only when given requirements are satisfied.

    .. admonition:: Example

       This test case runs only when `numpy>=1.10` is installed.

       >>> from cupy import testing
       ... class Test(unittest.TestCase):
       ...     @testing.with_requires('numpy>=1.10')
       ...     def test_for_numpy_1_10(self):
       ...         pass

    Args:
        requirements: A list of string representing requirement condition to
            run a given test case.

    """
    ws = pkg_resources.WorkingSet()
    try:
        ws.require(*requirements)
        skip = False
    except pkg_resources.ResolutionError:
        skip = True

    msg = 'requires: {}'.format(','.join(requirements))
    return unittest.skipIf(skip, msg)


def shaped_arange(shape, xp=cupy, dtype=numpy.float32):
    """Returns an array with given shape, array module, and dtype.

    Args:
         shape(tuple of int): Shape of returned ndarray.
         xp(numpy or cupy): Array module to use.
         dtype(dtype): Dtype of returned ndarray.

    Returns:
         numpy.ndarray or cupy.ndarray:
         The array filled with :math:`1, \\cdots, N` with specified dtype
         with given shape, array module. Here, :math:`N` is
         the size of the returned array.
         If ``dtype`` is ``numpy.bool_``, evens (resp. odds) are converted to
         ``True`` (resp. ``False``).

    """
    dtype = numpy.dtype(dtype)
    a = numpy.arange(1, internal.prod(shape) + 1, 1)
    if dtype == '?':
        a = a % 2 == 0
    elif dtype.kind == 'c':
        a = a + a * 1j
    return xp.array(a.astype(dtype).reshape(shape))


def shaped_reverse_arange(shape, xp=cupy, dtype=numpy.float32):
    """Returns an array filled with decreasing numbers.

    Args:
         shape(tuple of int): Shape of returned ndarray.
         xp(numpy or cupy): Array module to use.
         dtype(dtype): Dtype of returned ndarray.

    Returns:
         numpy.ndarray or cupy.ndarray:
         The array filled with :math:`N, \\cdots, 1` with specified dtype
         with given shape, array module.
         Here, :math:`N` is the size of the returned array.
         If ``dtype`` is ``numpy.bool_``, evens (resp. odds) are converted to
         ``True`` (resp. ``False``).
    """
    dtype = numpy.dtype(dtype)
    size = internal.prod(shape)
    a = numpy.arange(size, 0, -1)
    if dtype == '?':
        a = a % 2 == 0
    elif dtype.kind == 'c':
        a = a + a * 1j
    return xp.array(a.astype(dtype).reshape(shape))


def shaped_random(shape, xp=cupy, dtype=numpy.float32, scale=10, seed=0):
    """Returns an array filled with random values.

    Args:
         shape(tuple): Shape of returned ndarray.
         xp(numpy or cupy): Array module to use.
         dtype(dtype): Dtype of returned ndarray.
         scale(float): Scaling factor of elements.
         seed(int): Random seed.

    Returns:
         numpy.ndarray or cupy.ndarray: The array with
             given shape, array module,

    If ``dtype`` is ``numpy.bool_``, the elements are
    independently drawn from ``True`` and ``False``
    with same probabilities.
    Otherwise, the array is filled with samples
    independently and identically drawn
    from uniform distribution over :math:`[0, scale)`
    with specified dtype.
    """
    numpy.random.seed(seed)
    dtype = numpy.dtype(dtype)
    if dtype == '?':
        return xp.asarray(numpy.random.randint(2, size=shape).astype(dtype))
    elif dtype.kind == 'c':
        a = numpy.random.rand(*shape) + 1j * numpy.random.rand(*shape)
        return xp.asarray((a * scale).astype(dtype))
    else:
        return xp.asarray((numpy.random.rand(*shape) * scale).astype(dtype))


class NumpyError(object):

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.err = numpy.geterr()
        numpy.seterr(**self.kw)

    def __exit__(self, *_):
        numpy.seterr(**self.err)


@contextlib.contextmanager
def assert_warns(expected):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        yield

    if any(isinstance(m.message, expected) for m in w):
        return

    try:
        exc_name = expected.__name__
    except AttributeError:
        exc_name = str(expected)

    raise AssertionError('%s not triggerred' % exc_name)


class NumpyAliasTestBase(unittest.TestCase):

    @property
    def func(self):
        raise NotImplementedError()

    @property
    def cupy_func(self):
        return getattr(cupy, self.func)

    @property
    def numpy_func(self):
        return getattr(numpy, self.func)


class NumpyAliasBasicTestBase(NumpyAliasTestBase):

    def test_argspec(self):
        if six.PY2:
            f = inspect.getargspec
        else:
            f = inspect.signature
        self.assertEqual(
            f(self.cupy_func),
            f(self.numpy_func))

    def test_docstring(self):
        cupy_func = self.cupy_func
        numpy_func = self.numpy_func
        self.assertTrue(hasattr(cupy_func, '__doc__'))
        self.assertNotEqual(cupy_func.__doc__, None)
        self.assertNotEqual(cupy_func.__doc__, '')
        self.assertIsNot(cupy_func.__doc__, numpy_func.__doc__)


class NumpyAliasValuesTestBase(NumpyAliasTestBase):

    def test_values(self):
        self.assertEqual(
            self.cupy_func(*self.args),
            self.numpy_func(*self.args))
