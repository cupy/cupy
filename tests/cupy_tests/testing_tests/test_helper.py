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

    def test_both_success(self):
        with self.assertRaises(AssertionError):
            helper._check_cupy_numpy_error(self, None, None, None, None)

    def test_cupy_error(self):
        cupy_error = Exception()
        cupy_tb = 'xxxx'
        with six.assertRaisesRegex(self, AssertionError, cupy_tb):
            helper._check_cupy_numpy_error(self, cupy_error, cupy_tb,
                                           None, None)

    def test_numpy_error(self):
        numpy_error = Exception()
        numpy_tb = 'yyyy'
        with six.assertRaisesRegex(self, AssertionError, numpy_tb):
            helper._check_cupy_numpy_error(self, None, None,
                                           numpy_error, numpy_tb)

    def test_cupy_numpy_different_error(self):
        cupy_error = TypeError()
        cupy_tb = 'xxxx'
        numpy_error = ValueError()
        numpy_tb = 'yyyy'
        # Use re.S mode to ignore new line characters
        pattern = re.compile(cupy_tb + '.*' + numpy_tb, re.S)
        with six.assertRaisesRegex(self, AssertionError, pattern):
            helper._check_cupy_numpy_error(self, cupy_error, cupy_tb,
                                           numpy_error, numpy_tb)

    def test_same_error(self):
        cupy_error = Exception()
        cupy_tb = 'xxxx'
        numpy_error = Exception()
        numpy_tb = 'yyyy'
        # Nothing happens
        helper._check_cupy_numpy_error(self, cupy_error, cupy_tb,
                                       numpy_error, numpy_tb,
                                       accept_error=Exception)

    def test_forbidden_error(self):
        cupy_error = Exception()
        cupy_tb = 'xxxx'
        numpy_error = Exception()
        numpy_tb = 'yyyy'
        # Use re.S mode to ignore new line characters
        pattern = re.compile(cupy_tb + '.*' + numpy_tb, re.S)
        with six.assertRaisesRegex(self, AssertionError, pattern):
            helper._check_cupy_numpy_error(
                self, cupy_error, cupy_tb,
                numpy_error, numpy_tb, accept_error=False)


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

    def test_correct_failure(self):
        numpy.testing.assert_raises_regex(
            AssertionError, 'mismatch 100.0%', self.correct_failure)

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
