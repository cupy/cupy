import unittest
import warnings

import numpy
import pytest

import cupy
from cupy import testing


@testing.parameterize(
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': (1, 0, 2)},
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': (-1, 0, -2)},
    {'shape': (2, 3, 4), 'transpose': (2, 0, 1), 'indexes': (1, 0, 2)},
    {'shape': (2, 3, 4), 'transpose': (2, 0, 1), 'indexes': (-1, 0, -2)},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (slice(None), slice(None, 1), slice(2))},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (slice(None), slice(None, -1), slice(-2))},
    {'shape': (2, 3, 4), 'transpose': (2, 0, 1),
     'indexes': (slice(None), slice(None, 1), slice(2))},
    {'shape': (2, 3, 5), 'transpose': None,
     'indexes': (slice(None, None, -1), slice(1, None, -1), slice(4, 1, -2))},
    {'shape': (2, 3, 5), 'transpose': (2, 0, 1),
     'indexes': (slice(4, 1, -2), slice(None, None, -1), slice(1, None, -1))},
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': (Ellipsis, 2)},
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': (1, Ellipsis)},
    {'shape': (2, 3, 4, 5), 'transpose': None, 'indexes': (1, Ellipsis, 3)},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (1, None, slice(2), None, 2)},
    {'shape': (2, 3), 'transpose': None, 'indexes': (None,)},
    {'shape': (2,), 'transpose': None, 'indexes': (slice(None,), None)},
    {'shape': (), 'transpose': None, 'indexes': (None,)},
    {'shape': (), 'transpose': None, 'indexes': (None, None)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(10, -9, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-9, -10, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-1, -10, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-1, -11, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-11, -11, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(10, -9, -3),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-1, -11, -3),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(1, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(0, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-1, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-4, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-6, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-10, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-11, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-12, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, 1, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, 0, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -1, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -4, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -5, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -6, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -10, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -11, -1),)},
    {'shape': (10,), 'transpose': None, 'indexes': (slice(-5, -12, -1),)},
    # reversing indexing on empty dimension
    {'shape': (0,), 'transpose': None, 'indexes': (slice(None, None, -1),)},
    {'shape': (0, 0), 'transpose': None,
     'indexes': (slice(None, None, -1), slice(None, None, -1))},
    {'shape': (0, 0), 'transpose': None,
     'indexes': (None, slice(None, None, -1))},
    {'shape': (0, 0), 'transpose': None,
     'indexes': (slice(None, None, -1), None)},
    {'shape': (0, 1), 'transpose': None,
     'indexes': (slice(None, None, -1), None)},
    {'shape': (1, 0), 'transpose': None,
     'indexes': (None, slice(None, None, -1))},
    {'shape': (1, 0, 1), 'transpose': None,
     'indexes': (None, slice(None, None, -1), None)},
    #
    {'shape': (2, 0), 'transpose': None,
     'indexes': (1, slice(None, None, None))},
)
@testing.gpu
class TestArrayIndexingParameterized(unittest.TestCase):

    _getitem_hip_skip_condition = [
        ((1, 0, 2), (2, 3, 4), None),
        ((-1, 0, -2), (2, 3, 4), None),
        ((1, 0, 2), (2, 3, 4), (2, 0, 1)),
        ((-1, 0, -2), (2, 3, 4), (2, 0, 1)),
        ((slice(None, None, None), None), (2,), None),
        ((slice(-9, -10, -1),), (10,), None),
        ((slice(-4, -5, -1),), (10,), None),
        ((slice(-5, -6, -1),), (10,), None),
    ]

    def _check_getitem_hip_skip_condition(self):
        return (self.indexes, self.shape, self.transpose) in \
            self._getitem_hip_skip_condition

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_getitem(self, xp, dtype):
        if cupy.cuda.runtime.is_hip:
            if self._check_getitem_hip_skip_condition():
                pytest.xfail('HIP may have a bug')
        a = testing.shaped_arange(self.shape, xp, dtype)
        if self.transpose:
            a = a.transpose(self.transpose)
        return a[self.indexes]


@testing.parameterize(
    {'shape': (), 'transpose': None, 'indexes': 0},
    {'shape': (), 'transpose': None, 'indexes': (slice(0, 1, 0),)},
    {'shape': (2, 3), 'transpose': None, 'indexes': (0, 0, 0)},
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': -3},
    {'shape': (2, 3, 4), 'transpose': (2, 0, 1), 'indexes': -5},
    {'shape': (2, 3, 4), 'transpose': None, 'indexes': 3},
    {'shape': (2, 3, 4), 'transpose': (2, 0, 1), 'indexes': 5},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (slice(0, 1, 0), )},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (slice((0, 0), None, None), )},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (slice(None, (0, 0), None), )},
    {'shape': (2, 3, 4), 'transpose': None,
     'indexes': (slice(None, None, (0, 0)), )},
)
@testing.gpu
class TestArrayInvalidIndex(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_invalid_getitem(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp, dtype)
            if self.transpose:
                a = a.transpose(self.transpose)
            with pytest.raises((ValueError, IndexError, TypeError)):
                a[self.indexes]


@testing.gpu
class TestArrayIndex(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setitem_constant(self, xp, dtype):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        a[:] = 1
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setitem_partial_constant(self, xp, dtype):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        a[1, 1:3] = 1
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setitem_copy(self, xp, dtype):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        b = testing.shaped_arange((2, 3, 4), xp, dtype)
        a[:] = b
        return a

    @testing.for_all_dtypes_combination(('src_type', 'dst_type'))
    @testing.numpy_cupy_array_equal()
    def test_setitem_different_type(self, xp, src_type, dst_type):
        a = xp.zeros((2, 3, 4), dtype=dst_type)
        b = testing.shaped_arange((2, 3, 4), xp, src_type)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            a[:] = b
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setitem_partial_copy(self, xp, dtype):
        a = xp.zeros((2, 3, 4), dtype=dtype)
        b = testing.shaped_arange((3, 2), xp, dtype)
        a[1, ::-1, 1:4:2] = b
        return a

    @testing.numpy_cupy_array_equal()
    def test_T(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        return a.T

    @testing.numpy_cupy_array_equal()
    def test_T_vector(self, xp):
        a = testing.shaped_arange((4,), xp)
        return a.T
