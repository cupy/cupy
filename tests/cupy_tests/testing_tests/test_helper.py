import re
import unittest

import numpy

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
        with self.assertRaisesRegexp(AssertionError, cupy_tb):
            helper._check_cupy_numpy_error(self, cupy_error, cupy_tb,
                                           None, None)

    def test_numpy_error(self):
        numpy_error = Exception()
        numpy_tb = 'yyyy'
        with self.assertRaisesRegexp(AssertionError, numpy_tb):
            helper._check_cupy_numpy_error(self, None, None,
                                           numpy_error, numpy_tb)

    def test_cupy_numpy_different_error(self):
        cupy_error = TypeError()
        cupy_tb = 'xxxx'
        numpy_error = ValueError()
        numpy_tb = 'yyyy'
        # Use re.S mode to ignore new line characters
        pattern = re.compile(cupy_tb + '.*' + numpy_tb, re.S)
        with self.assertRaisesRegexp(AssertionError, pattern):
            helper._check_cupy_numpy_error(self, cupy_error, cupy_tb,
                                           numpy_error, numpy_tb)

    def test_same_error(self):
        cupy_error = Exception()
        cupy_tb = 'xxxx'
        numpy_error = Exception()
        numpy_tb = 'yyyy'
        # Nothing happens
        helper._check_cupy_numpy_error(self, cupy_error, cupy_tb,
                                       numpy_error, numpy_tb)

    def test_forbidden_error(self):
        cupy_error = Exception()
        cupy_tb = 'xxxx'
        numpy_error = Exception()
        numpy_tb = 'yyyy'
        # Use re.S mode to ignore new line characters
        pattern = re.compile(cupy_tb + '.*' + numpy_tb, re.S)
        with self.assertRaisesRegexp(AssertionError, pattern):
            helper._check_cupy_numpy_error(
                self, cupy_error, cupy_tb,
                numpy_error, numpy_tb, accept_error=False)


def make_func(np_result, cp_result):
    def f(self, xp):
        if xp == numpy:
            return np_result
        elif xp == cupy:
            return cp_result
    return f


def make_func_foo(np_result, cp_result):
    def f(self, foo):
        if foo == numpy:
            return np_result
        elif foo == cupy:
            return cp_result
    return f


class NumPyCuPyDecoratorBase(object):

    def test_valid(self):
        decorator = getattr(testing, self.decorator)()
        decorated_func = decorator(self.valid_func)
        decorated_func(self)

    def test_invalid(self):
        decorator = getattr(testing, self.decorator)()
        decorated_func = decorator(self.invalid_func)
        with self.assertRaises(AssertionError):
            decorated_func(self)

    def test_name(self):
        decorator = getattr(testing, self.decorator)(name='foo')
        decorated_func = decorator(self.strange_kw_func)
        decorated_func(self)


def numpy_error(self, xp):
    if xp == numpy:
        raise ValueError()
    elif xp == cupy:
        return cupy.array(1)


def cupy_error(self, xp):
    if xp == numpy:
        return numpy.array(1)
    elif xp == cupy:
        raise ValueError()


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


@testing.parameterize(
    {'decorator': 'numpy_cupy_allclose'},
    {'decorator': 'numpy_cupy_array_almost_equal'},
    {'decorator': 'numpy_cupy_array_almost_equal_nulp'},
    {'decorator': 'numpy_cupy_array_max_ulp'},
    {'decorator': 'numpy_cupy_array_equal'}
)
class TestNumPyCuPyEqual(unittest.TestCase, NumPyCuPyDecoratorBase,
                         NumPyCuPyDecoratorBase2):

    def setUp(self):
        self.valid_func = make_func(numpy.array(1), cupy.array(1))
        self.invalid_func = make_func(numpy.array(1), cupy.array(2))
        self.strange_kw_func = make_func_foo(numpy.array(1), cupy.array(1))


@testing.parameterize(
    {'decorator': 'numpy_cupy_array_list_equal'}
)
class TestNumPyCuPyListEqual(unittest.TestCase, NumPyCuPyDecoratorBase):

    def setUp(self):
        self.valid_func = make_func([numpy.array(1)], [cupy.array(1)])
        self.invalid_func = make_func([numpy.array(1)], [cupy.array(2)])
        self.strange_kw_func = make_func_foo([numpy.array(1)], [cupy.array(1)])


@testing.parameterize(
    {'decorator': 'numpy_cupy_array_less'}
)
class TestNumPyCuPyLess(unittest.TestCase, NumPyCuPyDecoratorBase,
                        NumPyCuPyDecoratorBase2):

    def setUp(self):
        self.valid_func = make_func(numpy.array(2), cupy.array(1))
        self.invalid_func = make_func(numpy.array(1), cupy.array(2))
        self.strange_kw_func = make_func_foo(
            numpy.array(2), cupy.array(1))


def numpy_cupy_raise(self, xp):
    raise ValueError()


def numpy_cupy_raise_foo(self, foo):
    raise ValueError()


@testing.parameterize(
    {'decorator': 'numpy_cupy_raises'}
)
class TestNumPyCuPyRaise(unittest.TestCase, NumPyCuPyDecoratorBase):

    def setUp(self):
        self.valid_func = numpy_cupy_raise
        self.invalid_func = make_func(numpy.array(1), cupy.array(1))
        self.strange_kw_func = numpy_cupy_raise_foo

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
