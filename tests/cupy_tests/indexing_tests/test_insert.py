import unittest

import numpy
import pytest

from cupy import testing


@testing.parameterize(*testing.product({
    'shape': [(7,), (2, 3), (4, 3, 2)],
    'n_vals': [0, 1, 3, 15],
}))
@testing.gpu
class TestPlace(unittest.TestCase):

    # NumPy 1.9 don't wraps values.
    # https://github.com/numpy/numpy/pull/5821
    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_place(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        if self.n_vals == 0:
            mask = xp.zeros(self.shape, dtype=numpy.bool_)
        else:
            mask = testing.shaped_random(self.shape, xp, numpy.bool_)
        vals = testing.shaped_random((self.n_vals,), xp, dtype)
        xp.place(a, mask, vals)
        return a


@testing.parameterize(*testing.product({
    'shape': [(7,), (2, 3), (4, 3, 2)],
}))
@testing.gpu
class TestPlaceRaises(unittest.TestCase):

    # NumPy 1.9 performs illegal memory access.
    # https://github.com/numpy/numpy/pull/5821
    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises(accept_error=ValueError)
    def test_place_empty_value_error(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        mask = testing.shaped_arange(self.shape, xp, numpy.int) % 2 == 0
        vals = testing.shaped_random((0,), xp, dtype)
        xp.place(a, mask, vals)

    # Before NumPy 1.12 it was TypeError.
    # https://github.com/numpy/numpy/pull/7003
    @testing.with_requires('numpy>=1.12')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises(accept_error=ValueError)
    def test_place_shape_unmatch_error(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        mask = testing.shaped_random((3, 4), xp, numpy.bool_)
        vals = testing.shaped_random((1,), xp, dtype)
        xp.place(a, mask, vals)


@testing.parameterize(*testing.product({
    'shape': [(7,), (2, 3), (4, 3, 2)],
    'mode': ['raise', 'wrap', 'clip'],
    'n_vals': [0, 1, 3, 4, 5],
}))
@testing.gpu
class TestPut(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_put(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        # Take care so that actual indices don't overlap.
        if self.mode == 'raise':
            inds = xp.array([2, -1, 3, 0])
        else:
            inds = xp.array([2, -8, 3, 7])
        vals = testing.shaped_random((self.n_vals,), xp, dtype)
        xp.put(a, inds, vals, self.mode)
        return a


@testing.parameterize(*testing.product({
    'shape': [(7,), (2, 3)],
}))
@testing.gpu
class TestPutRaises(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises(accept_error=IndexError)
    def test_put_inds_underflow_error(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        inds = xp.array([2, -8, 3, 0])
        vals = testing.shaped_random((4,), xp, dtype)
        xp.put(a, inds, vals, mode='raise')

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises(accept_error=IndexError)
    def test_put_inds_overflow_error(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        inds = xp.array([2, -1, 3, 7])
        vals = testing.shaped_random((4,), xp, dtype)
        xp.put(a, inds, vals, mode='raise')

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises(accept_error=TypeError)
    def test_put_mode_error(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        inds = xp.array([2, -1, 3, 0])
        vals = testing.shaped_random((4,), xp, dtype)
        xp.put(a, inds, vals, mode='unknown')


@testing.parameterize(*testing.product({
    'shape': [(3, 3), (2, 2, 2), (3, 5), (5, 3)],
    'val': [1, 0, (2,), (2, 2)],
    'wrap': [True, False],
}))
@testing.gpu
class TestFillDiagonal(unittest.TestCase):

    def _compute_val(self, xp):
        if type(self.val) is int:
            return self.val
        else:
            return xp.arange(numpy.prod(self.val)).reshape(self.val)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_fill_diagonal(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        val = self._compute_val(xp)
        xp.fill_diagonal(a, val=val, wrap=self.wrap)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_columnar_slice(self, xp, dtype):  # see cupy#2970
        if self.shape == (2, 2, 2):
            pytest.skip(
                'The length of each dimension must be the same after slicing')
        a = testing.shaped_arange(self.shape, xp, dtype)
        val = self._compute_val(xp)
        xp.fill_diagonal(a[:, 1:], val=val, wrap=self.wrap)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_1darray(self, xp, dtype):
        a = testing.shaped_arange(5, xp, dtype)
        val = self._compute_val(xp)
        xp.fill_diagonal(a, val=val, wrap=self.wrap)
