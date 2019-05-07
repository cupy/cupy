import re
import unittest

import numpy
import six

import cupy
from cupy import testing
from cupy.testing import helper


class TestContainsSignedAndUnsigned(unittest.TestCase):

    def test_include(self):
        kw = {'x': numpy.int32, 'y': numpy.uint32}
        self.assertTrue(helper._contains_signed_and_unsigned(kw))

        kw = {'x': numpy.float32, 'y': numpy.uint32}
        self.assertTrue(helper._contains_signed_and_unsigned(kw))

    def test_signed_only(self):
        kw = {'x': numpy.int32}
        self.assertFalse(helper._contains_signed_and_unsigned(kw))

        kw = {'x': numpy.float}
        self.assertFalse(helper._contains_signed_and_unsigned(kw))

    def test_unsigned_only(self):
        kw = {'x': numpy.uint32}
        self.assertFalse(helper._contains_signed_and_unsigned(kw))


class TestCheckCupyNumpyError(unittest.TestCase):

    tbs = {
        cupy: 'xxxx',
        numpy: 'yyyy'
    }

    def test_both_success(self):
        @testing.helper.numpy_cupy_raises()
        def dummy_both_success(self, xp):
            pass

        with self.assertRaises(AssertionError):
            dummy_both_success(self)

    def test_cupy_error(self):
        @testing.helper.numpy_cupy_raises()
        def dummy_cupy_error(self, xp):
            if xp is cupy:
                raise Exception(self.tbs.get(cupy))

        with six.assertRaisesRegex(self, AssertionError, self.tbs.get(cupy)):
            dummy_cupy_error(self)

    def test_numpy_error(self):
        @testing.helper.numpy_cupy_raises()
        def dummy_numpy_error(self, xp):
            if xp is numpy:
                raise Exception(self.tbs.get(numpy))

        with six.assertRaisesRegex(self, AssertionError, self.tbs.get(numpy)):
            dummy_numpy_error(self)

    def test_cupy_numpy_different_error(self):
        @testing.helper.numpy_cupy_raises()
        def dummy_cupy_numpy_different_error(self, xp):
            if xp is cupy:
                raise TypeError(self.tbs.get(cupy))
            elif xp is numpy:
                raise ValueError(self.tbs.get(numpy))

        # Use re.S mode to ignore new line characters
        pattern = re.compile(
            self.tbs.get(cupy) + '.*' + self.tbs.get(numpy), re.S)
        with six.assertRaisesRegex(self, AssertionError, pattern):
            dummy_cupy_numpy_different_error(self)

    def test_cupy_derived_error(self):
        @testing.helper.numpy_cupy_raises()
        def dummy_cupy_derived_error(self, xp):
            if xp is cupy:
                raise ValueError(self.tbs.get(cupy))
            elif xp is numpy:
                raise Exception(self.tbs.get(numpy))

        dummy_cupy_derived_error(self)  # Assert no exceptions

    def test_numpy_derived_error(self):
        @testing.helper.numpy_cupy_raises()
        def dummy_numpy_derived_error(self, xp):
            if xp is cupy:
                raise Exception(self.tbs.get(cupy))
            elif xp is numpy:
                raise IndexError(self.tbs.get(numpy))

        # NumPy errors may not derive from CuPy errors, i.e. CuPy errors should
        # be at least as explicit as the NumPy error
        pattern = re.compile(
            self.tbs.get(cupy) + '.*' + self.tbs.get(numpy), re.S)
        with six.assertRaisesRegex(self, AssertionError, pattern):
            dummy_numpy_derived_error(self)

    def test_same_error(self):
        @testing.helper.numpy_cupy_raises(accept_error=Exception)
        def dummy_same_error(self, xp):
            raise Exception(self.tbs.get(xp))

        dummy_same_error(self)

    def test_cupy_derived_unaccept_error(self):
        @testing.helper.numpy_cupy_raises(accept_error=ValueError)
        def dummy_cupy_derived_unaccept_error(self, xp):
            if xp is cupy:
                raise IndexError(self.tbs.get(cupy))
            elif xp is numpy:
                raise Exception(self.tbs.get(numpy))

        # Neither `IndexError` nor `Exception` is derived from `ValueError`,
        # therefore expect an error
        pattern = re.compile(
            self.tbs.get(cupy) + '.*' + self.tbs.get(numpy), re.S)
        with six.assertRaisesRegex(self, AssertionError, pattern):
            dummy_cupy_derived_unaccept_error(self)

    def test_numpy_derived_unaccept_error(self):
        @testing.helper.numpy_cupy_raises(accept_error=ValueError)
        def dummy_numpy_derived_unaccept_error(self, xp):
            if xp is cupy:
                raise Exception(self.tbs.get(cupy))
            elif xp is numpy:
                raise ValueError(self.tbs.get(numpy))

        # `Exception` is not derived from `ValueError`, therefore expect an
        # error
        pattern = re.compile(
            self.tbs.get(cupy) + '.*' + self.tbs.get(numpy), re.S)
        with six.assertRaisesRegex(self, AssertionError, pattern):
            dummy_numpy_derived_unaccept_error(self)

    def test_forbidden_error(self):
        @testing.helper.numpy_cupy_raises(accept_error=False)
        def dummy_forbidden_error(self, xp):
            raise Exception(self.tbs.get(xp))

        pattern = re.compile(
            self.tbs.get(cupy) + '.*' + self.tbs.get(numpy), re.S)
        with six.assertRaisesRegex(self, AssertionError, pattern):
            dummy_forbidden_error(self)


class NumPyCuPyDecoratorBase(object):

    def test_valid(self):
        decorator = getattr(testing, self.decorator)()
        decorated_func = decorator(type(self).valid_func)
        decorated_func(self)

    def test_invalid(self):
        decorator = getattr(testing, self.decorator)()
        decorated_func = decorator(type(self).invalid_func)
        with self.assertRaises(AssertionError):
            decorated_func(self)

    def test_name(self):
        decorator = getattr(testing, self.decorator)(name='foo')
        decorated_func = decorator(type(self).strange_kw_func)
        decorated_func(self)


def numpy_error(_, xp):
    if xp == numpy:
        raise ValueError()
    elif xp == cupy:
        return cupy.array(1)


def cupy_error(_, xp):
    if xp == numpy:
        return numpy.array(1)
    elif xp == cupy:
        raise ValueError()


@testing.gpu
class NumPyCuPyDecoratorBase2(object):

    def test_accept_error_numpy(self):
        decorator = getattr(testing, self.decorator)(accept_error=False)
        decorated_func = decorator(numpy_error)
        with self.assertRaises(AssertionError):
            decorated_func(self)

    def test_accept_error_cupy(self):
        decorator = getattr(testing, self.decorator)(accept_error=False)
        decorated_func = decorator(cupy_error)
        with self.assertRaises(AssertionError):
            decorated_func(self)


def make_result(xp, np_result, cp_result):
    if xp == numpy:
        return np_result
    elif xp == cupy:
        return cp_result


@testing.parameterize(
    {'decorator': 'numpy_cupy_allclose'},
    {'decorator': 'numpy_cupy_array_almost_equal'},
    {'decorator': 'numpy_cupy_array_almost_equal_nulp'},
    {'decorator': 'numpy_cupy_array_max_ulp'},
    {'decorator': 'numpy_cupy_array_equal'}
)
class TestNumPyCuPyEqual(unittest.TestCase, NumPyCuPyDecoratorBase,
                         NumPyCuPyDecoratorBase2):

    def valid_func(self, xp):
        return make_result(xp, numpy.array(1), cupy.array(1))

    def invalid_func(self, xp):
        return make_result(xp, numpy.array(1), cupy.array(2))

    def strange_kw_func(self, foo):
        return make_result(foo, numpy.array(1), cupy.array(1))


@testing.parameterize(
    {'decorator': 'numpy_cupy_array_list_equal'}
)
@testing.gpu
class TestNumPyCuPyListEqual(unittest.TestCase, NumPyCuPyDecoratorBase):

    def valid_func(self, xp):
        return make_result(xp, [numpy.array(1)], [cupy.array(1)])

    def invalid_func(self, xp):
        return make_result(xp, [numpy.array(1)], [cupy.array(2)])

    def strange_kw_func(self, foo):
        return make_result(foo, [numpy.array(1)], [cupy.array(1)])


@testing.parameterize(
    {'decorator': 'numpy_cupy_array_less'}
)
class TestNumPyCuPyLess(unittest.TestCase, NumPyCuPyDecoratorBase,
                        NumPyCuPyDecoratorBase2):

    def valid_func(self, xp):
        return make_result(xp, numpy.array(2), cupy.array(1))

    def invalid_func(self, xp):
        return make_result(xp, numpy.array(1), cupy.array(2))

    def strange_kw_func(self, foo):
        return make_result(foo, numpy.array(2), cupy.array(1))


@testing.parameterize(
    {'decorator': 'numpy_cupy_raises'}
)
class TestNumPyCuPyRaise(unittest.TestCase, NumPyCuPyDecoratorBase):

    def valid_func(self, xp):
        raise ValueError()

    def invalid_func(self, xp):
        return make_result(xp, numpy.array(1), cupy.array(1))

    def strange_kw_func(self, foo):
        raise ValueError()

    def test_accept_error_numpy(self):
        decorator = getattr(testing, self.decorator)()
        decorated_func = decorator(numpy_error)
        with self.assertRaises(AssertionError):
            decorated_func(self)

    def test_accept_error_cupy(self):
        decorator = getattr(testing, self.decorator)()
        decorated_func = decorator(cupy_error)
        with self.assertRaises(AssertionError):
            decorated_func(self)


class TestIgnoreOfNegativeValueDifferenceOnCpuAndGpu(unittest.TestCase):

    @helper.for_unsigned_dtypes('dtype1')
    @helper.for_signed_dtypes('dtype2')
    @helper.numpy_cupy_allclose()
    def correct_failure(self, xp, dtype1, dtype2):
        if xp == numpy:
            return xp.array(-1, dtype=numpy.float32)
        else:
            return xp.array(-2, dtype=numpy.float32)

    @testing.with_requires('numpy>=1.16.1')
    def test_correct_failure(self):
        with six.assertRaisesRegex(self, AssertionError, 'Mismatch: 100%'):
            self.correct_failure()

    @testing.with_requires('numpy<1.16.1')
    def test_correct_failure_old_np(self):
        with six.assertRaisesRegex(self, AssertionError, 'mismatch 100\\.0%'):
            self.correct_failure()

    @helper.for_unsigned_dtypes('dtype1')
    @helper.for_signed_dtypes('dtype2')
    @helper.numpy_cupy_allclose()
    def test_correct_success(self, xp, dtype1, dtype2):
        # Behavior of assigning a negative value to an unsigned integer
        # variable is undefined.
        # nVidia GPUs and Intel CPUs behave differently.
        # To avoid this difference, we need to ignore dimensions whose
        # values are negative.
        if xp == numpy:
            return xp.array(-1, dtype=dtype1)
        else:
            return xp.array(-2, dtype=dtype1)


@testing.parameterize(
    {'xp': numpy},
    {'xp': cupy},
)
@testing.gpu
class TestShapedRandom(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_shape_and_dtype(self, dtype):
        a = testing.shaped_random((2, 3), self.xp, dtype)
        self.assertTrue(a.shape == (2, 3))
        self.assertTrue(a.dtype == dtype)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    def test_value_range(self, dtype):
        a = testing.shaped_random((2, 3), self.xp, dtype)
        self.assertTrue(self.xp.all(0 <= a))
        self.assertTrue(self.xp.all(a < 10))

    def test_bool(self):
        a = testing.shaped_random(10000, self.xp, numpy.bool_)
        self.assertTrue(4000 < self.xp.sum(a) < 6000)

    @testing.for_complex_dtypes()
    def test_complex(self, dtype):
        a = testing.shaped_random((2, 3), self.xp, dtype)
        self.assertTrue(self.xp.all(0 <= a.real))
        self.assertTrue(self.xp.all(a.real < 10))
        self.assertTrue(self.xp.all(0 <= a.imag))
        self.assertTrue(self.xp.all(a.imag < 10))
        self.assertTrue(self.xp.any(a.imag))
