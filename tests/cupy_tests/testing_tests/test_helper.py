import re
import unittest

import numpy
import pytest
import six

import cupy
from cupy import testing
from cupy.testing import helper


class _Exception1(Exception):
    pass


class _Exception2(Exception):
    pass


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
                raise _Exception1(self.tbs.get(cupy))
            elif xp is numpy:
                raise _Exception2(self.tbs.get(numpy))

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

    def test_axis_error_different_type(self):
        @testing.helper.numpy_cupy_raises()
        def dummy_axis_error(self, xp):
            if xp is cupy:
                raise cupy.core._errors._AxisError(self.tbs.get(cupy))
            elif xp is numpy:
                raise TypeError(self.tbs.get(numpy))

        pattern = re.compile(
            self.tbs.get(cupy) + '.*' + self.tbs.get(numpy), re.S)
        with six.assertRaisesRegex(self, AssertionError, pattern):
            dummy_axis_error(self)

    @testing.with_requires('numpy>=1.13')
    def test_axis_error_value_different_type(self):
        @testing.helper.numpy_cupy_raises()
        def dummy_axis_error(self, xp):
            if xp is cupy:
                raise cupy.core._errors._AxisError(self.tbs.get(cupy))
            elif xp is numpy:
                raise ValueError(self.tbs.get(numpy))

        pattern = re.compile(
            self.tbs.get(cupy) + '.*' + self.tbs.get(numpy), re.S)
        with six.assertRaisesRegex(self, AssertionError, pattern):
            dummy_axis_error(self)

    @testing.with_requires('numpy>=1.13')
    def test_axis_error_index_different_type(self):
        @testing.helper.numpy_cupy_raises()
        def dummy_axis_error(self, xp):
            if xp is cupy:
                raise cupy.core._errors._AxisError(self.tbs.get(cupy))
            elif xp is numpy:
                raise IndexError(self.tbs.get(numpy))

        pattern = re.compile(
            self.tbs.get(cupy) + '.*' + self.tbs.get(numpy), re.S)
        with six.assertRaisesRegex(self, AssertionError, pattern):
            dummy_axis_error(self)

    @testing.with_requires('numpy<1.13')
    def test_axis_error_value(self):
        @testing.helper.numpy_cupy_raises()
        def dummy_axis_error(self, xp):
            if xp is cupy:
                raise cupy.core._errors._AxisError(self.tbs.get(cupy))
            elif xp is numpy:
                raise ValueError(self.tbs.get(numpy))

        dummy_axis_error(self)

    @testing.with_requires('numpy<1.13')
    def test_axis_error_index(self):
        @testing.helper.numpy_cupy_raises()
        def dummy_axis_error(self, xp):
            if xp is cupy:
                raise cupy.core._errors._AxisError(self.tbs.get(cupy))
            elif xp is numpy:
                raise IndexError(self.tbs.get(numpy))

        dummy_axis_error(self)


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
        with pytest.raises(AssertionError):
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


@testing.parameterize(*testing.product({
    'xp': [numpy, cupy],
    'shape': [(3, 2), (), (3, 0, 2)],
}))
@testing.gpu
class TestShapedRandom(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_shape_and_dtype(self, dtype):
        a = testing.shaped_random(self.shape, self.xp, dtype)
        self.assertIsInstance(a, self.xp.ndarray)
        self.assertTrue(a.shape == self.shape)
        self.assertTrue(a.dtype == dtype)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    def test_value_range(self, dtype):
        a = testing.shaped_random(self.shape, self.xp, dtype)
        self.assertTrue(self.xp.all(0 <= a))
        self.assertTrue(self.xp.all(a < 10))

    @testing.for_complex_dtypes()
    def test_complex(self, dtype):
        a = testing.shaped_random(self.shape, self.xp, dtype)
        self.assertTrue(self.xp.all(0 <= a.real))
        self.assertTrue(self.xp.all(a.real < 10))
        self.assertTrue(self.xp.all(0 <= a.imag))
        self.assertTrue(self.xp.all(a.imag < 10))
        if 0 not in self.shape:
            self.assertTrue(self.xp.any(a.imag))


@testing.parameterize(*testing.product({
    'xp': [numpy, cupy],
}))
@testing.gpu
class TestShapedRandomBool(unittest.TestCase):

    def test_bool(self):
        a = testing.shaped_random(10000, self.xp, numpy.bool_)
        self.assertTrue(4000 < self.xp.sum(a) < 6000)


class TestSkip(unittest.TestCase):

    @testing.numpy_cupy_allclose()
    def test_allclose(self, xp):
        raise unittest.SkipTest('Test for skip with @numpy_cupy_allclose')
        assert False

    @testing.numpy_cupy_array_almost_equal()
    def test_array_almost_equal(self, xp):
        raise unittest.SkipTest(
            'Test for skip with @numpy_cupy_array_almost_equal')
        assert False

    @testing.numpy_cupy_array_almost_equal_nulp()
    def test_array_almost_equal_nulp(self, xp):
        raise unittest.SkipTest(
            'Test for skip with @numpy_cupy_array_almost_equal_nulp')
        assert False

    @testing.numpy_cupy_array_max_ulp()
    def test_array_max_ulp(self, xp):
        raise unittest.SkipTest('Test for skip with @numpy_cupy_array_max_ulp')
        assert False

    @testing.numpy_cupy_array_equal()
    def test_array_equal(self, xp):
        raise unittest.SkipTest('Test for skip with @numpy_cupy_array_equal')
        assert False

    @testing.numpy_cupy_array_list_equal()
    def test_array_list_equal(self, xp):
        raise unittest.SkipTest(
            'Test for skip with @numpy_cupy_array_list_equal')
        assert False

    @testing.numpy_cupy_array_less()
    def test_less(self, xp):
        raise unittest.SkipTest('Test for skip with @numpy_cupy_array_less')
        assert False

    @testing.numpy_cupy_equal()
    def test_equal(self, xp):
        raise unittest.SkipTest('Test for skip with @numpy_cupy_equal')
        assert False

    @testing.numpy_cupy_raises()
    def test_raises(self, xp):
        raise unittest.SkipTest('Test for skip with @numpy_cupy_raises')
        assert False


class TestSkipFail(unittest.TestCase):

    @pytest.mark.xfail(strict=True)
    @testing.numpy_cupy_allclose()
    def test_different_reason(self, xp):
        if xp is numpy:
            raise unittest.SkipTest('skip1')
        else:
            raise unittest.SkipTest('skip2')

    @pytest.mark.xfail(strict=True)
    @testing.numpy_cupy_allclose()
    def test_only_numpy(self, xp):
        if xp is numpy:
            raise unittest.SkipTest('skip')
        else:
            return xp.array(True)

    @pytest.mark.xfail(strict=True)
    @testing.numpy_cupy_allclose()
    def test_only_cupy(self, xp):
        if xp is numpy:
            return xp.array(True)
        else:
            raise unittest.SkipTest('skip')
