import unittest
import warnings

import numpy
import pytest

import cupy
from cupy import testing


@testing.gpu
class TestFlatiter(unittest.TestCase):

    def test_base(self):
        for xp in (numpy, cupy):
            a = xp.zeros((2, 3, 4))
            assert a.flat.base is a

    def test_iter(self):
        for xp in (numpy, cupy):
            it = xp.zeros((2, 3, 4)).flat
            assert iter(it) is it

    def test_next(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            e = a.flatten()
            for ai, ei in zip(a.flat, e):
                assert ai == ei

    def test_len(self):
        for xp in (numpy, cupy):
            a = xp.zeros((2, 3, 4))
            assert len(a.flat) == 24
            assert len(a[::2].flat) == 12

    @testing.numpy_cupy_array_equal()
    def test_copy(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        o = a.flat.copy()
        assert a is not o
        return a.flat.copy()

    @testing.numpy_cupy_array_equal()
    def test_copy_next(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        it = a.flat
        it.__next__()
        return it.copy()  # Returns the flattened copy of whole `a`


@testing.parameterize(
    {'shape': (2, 3, 4), 'index': Ellipsis},
    {'shape': (2, 3, 4), 'index': 0},
    {'shape': (2, 3, 4), 'index': 10},
    {'shape': (2, 3, 4), 'index': slice(None)},
    {'shape': (2, 3, 4), 'index': slice(None, 10)},
    {'shape': (2, 3, 4), 'index': slice(None, None, 2)},
    {'shape': (2, 3, 4), 'index': slice(None, None, -1)},
    {'shape': (2, 3, 4), 'index': slice(10, None, -1)},
    {'shape': (2, 3, 4), 'index': slice(10, None, -2)},
    {'shape': (), 'index': slice(None)},
    {'shape': (10,), 'index': slice(None)},
)
@testing.gpu
class TestFlatiterSubscript(unittest.TestCase):

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_getitem(self, xp, dtype, order):
        a = testing.shaped_arange(self.shape, xp, dtype, order)
        return a.flat[self.index]

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setitem_scalar(self, xp, dtype, order):
        a = xp.zeros(self.shape, dtype=dtype, order=order)
        a.flat[self.index] = 1
        return a

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setitem_ndarray_1d(self, xp, dtype, order):
        if numpy.isscalar(self.index):
            pytest.skip()
        a = xp.zeros(self.shape, dtype=dtype, order=order)
        v = testing.shaped_arange((3,), xp, dtype, order)
        a.flat[self.index] = v
        return a

    @testing.for_CF_orders()
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_setitem_ndarray_nd(self, xp, dtype, order):
        if numpy.isscalar(self.index):
            pytest.skip()
        a = xp.zeros(self.shape, dtype=dtype, order=order)
        v = testing.shaped_arange((2, 3), xp, dtype, order)
        a.flat[self.index] = v
        return a

    @testing.for_CF_orders()
    @testing.for_all_dtypes_combination(('a_dtype', 'v_dtype'))
    @testing.numpy_cupy_array_equal()
    def test_setitem_ndarray_different_types(
            self, xp, a_dtype, v_dtype, order):
        if numpy.isscalar(self.index):
            pytest.skip()
        a = xp.zeros(self.shape, dtype=a_dtype, order=order)
        v = testing.shaped_arange((3,), xp, v_dtype, order)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', numpy.ComplexWarning)
            a.flat[self.index] = v
        return a


@testing.parameterize(
    {'shape': (2, 3, 4), 'index': None},
    {'shape': (2, 3, 4), 'index': (0,)},
    {'shape': (2, 3, 4), 'index': True},
    {'shape': (2, 3, 4), 'index': cupy.array([0])},
    {'shape': (2, 3, 4), 'index': [0]},
)
@testing.gpu
class TestFlatiterSubscriptIndexError(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_getitem(self, dtype):
        a = testing.shaped_arange(self.shape, cupy, dtype)
        with pytest.raises(IndexError):
            a.flat[self.index]

    @testing.for_all_dtypes()
    def test_setitem(self, dtype):
        a = testing.shaped_arange(self.shape, cupy, dtype)
        v = testing.shaped_arange((1,), cupy, dtype)
        with pytest.raises(IndexError):
            a.flat[self.index] = v
