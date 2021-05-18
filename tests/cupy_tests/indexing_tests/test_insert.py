import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.parameterize(*testing.product({
    'shape': [(7,), (2, 3), (4, 3, 2)],
    'n_vals': [0, 1, 3, 15],
}))
@testing.gpu
class TestPlace(unittest.TestCase):

    # NumPy 1.9 don't wraps values.
    # https://github.com/numpy/numpy/pull/5821
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
    def test_place_empty_value_error(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp, dtype)
            mask = testing.shaped_arange(self.shape, xp, int) % 2 == 0
            vals = testing.shaped_random((0,), xp, dtype)
            with pytest.raises(ValueError):
                xp.place(a, mask, vals)

    # Before NumPy 1.12 it was TypeError.
    # https://github.com/numpy/numpy/pull/7003
    @testing.for_all_dtypes()
    def test_place_shape_unmatch_error(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp, dtype)
            mask = testing.shaped_random((3, 4), xp, numpy.bool_)
            vals = testing.shaped_random((1,), xp, dtype)
            with pytest.raises(ValueError):
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
    'shape': [(7,), (2, 3), (4, 3, 2)],
}))
@testing.gpu
class TestPutScalars(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_put_index_scalar(self, xp):
        dtype = cupy.float32
        a = testing.shaped_arange(self.shape, xp, dtype)
        inds = 4
        vals = testing.shaped_random((4,), xp, dtype)
        xp.put(a, inds, vals)
        return a

    @testing.numpy_cupy_array_equal()
    def test_put_values_scalar(self, xp):
        dtype = cupy.float32
        a = testing.shaped_arange(self.shape, xp, dtype)
        # Take care so that actual indices don't overlap.
        inds = xp.array([2, 3, 5])
        vals = 3.0
        xp.put(a, inds, vals)
        return a


@testing.parameterize(*testing.product({
    'shape': [(7,), (2, 3)],
}))
@testing.gpu
class TestPutRaises(unittest.TestCase):

    @testing.for_all_dtypes()
    def test_put_inds_underflow_error(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp, dtype)
            inds = xp.array([2, -8, 3, 0])
            vals = testing.shaped_random((4,), xp, dtype)
            with pytest.raises(IndexError):
                xp.put(a, inds, vals, mode='raise')

    @testing.for_all_dtypes()
    def test_put_inds_overflow_error(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp, dtype)
            inds = xp.array([2, -1, 3, 7])
            vals = testing.shaped_random((4,), xp, dtype)
            with pytest.raises(IndexError):
                xp.put(a, inds, vals, mode='raise')

    @testing.with_requires('numpy>=1.19')
    @testing.for_all_dtypes()
    def test_put_mode_error(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange(self.shape, xp, dtype)
            inds = xp.array([2, -1, 3, 0])
            vals = testing.shaped_random((4,), xp, dtype)
            with pytest.raises(ValueError):
                xp.put(a, inds, vals, mode='unknown')


@testing.parameterize(
    *testing.product(
        {'shape': [(0,), (1,), (2, 3), (2, 3, 4)]}))
@testing.gpu
class TestPutmaskSameShape(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_putmask(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype=dtype, seed=0)
        mask = testing.shaped_random(self.shape, xp, dtype=numpy.bool_, seed=1)
        values = testing.shaped_random(self.shape, xp, dtype=dtype, seed=2)
        ret = xp.putmask(a, mask, values)
        assert ret is None
        return a


@testing.parameterize(
    *testing.product(
        {'shape': [(0,), (1,), (2, 3), (2, 3, 4)],
         'values_shape': [(2,), (3, 1), (5,)]}))
@testing.gpu
class TestPutmaskDifferentShapes(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_putmask(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype=dtype, seed=3)
        mask = testing.shaped_random(self.shape, xp, dtype=numpy.bool_, seed=4)
        values = testing.shaped_random(self.values_shape,
                                       xp, dtype=dtype, seed=5)
        ret = xp.putmask(a, mask, values)
        assert ret is None
        return a


@testing.gpu
class TestPutmask(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_putmask_scalar_values(self, xp):
        shape = (2, 3)
        a = testing.shaped_arange(shape, xp)
        xp.putmask(a, a > 1, 30)
        return a

    def test_putmask_non_equal_shape_raises(self):
        for xp in (numpy, cupy):
            a = xp.array([1, 2, 3])
            mask = xp.array([True, False])
            with pytest.raises(ValueError):
                xp.putmask(a, mask, a**2)

    @testing.numpy_cupy_array_equal()
    def test_putmask_int_mask_scalar_values(self, xp):
        a = xp.array([1, 2, 3, 3])
        mask = xp.array([0, 1, 0, 2])
        xp.putmask(a, mask, 0)
        return a


class TestPutmaskDifferentDtypes(unittest.TestCase):

    @testing.for_all_dtypes_combination(names=['a_dtype', 'val_dtype'])
    def test_putmask_differnt_dtypes_raises(self, a_dtype, val_dtype):
        shape = (2, 3)
        for xp in (numpy, cupy):
            a = testing.shaped_random(shape, xp, dtype=a_dtype)
            mask = testing.shaped_random(shape, xp, dtype=numpy.bool_)
            values = testing.shaped_random((3,), xp, dtype=val_dtype)
            if not numpy.can_cast(val_dtype, a_dtype):
                with pytest.raises(TypeError):
                    xp.putmask(a, mask, values)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_putmask_differnt_dtypes_mask(self, xp, dtype):
        shape = (2, 3)
        a = testing.shaped_random(shape, xp, dtype=numpy.int64)
        mask = testing.shaped_random(shape, xp, dtype=dtype)
        values = testing.shaped_random((3,), xp, dtype=numpy.int64)
        xp.putmask(a, mask, values)
        return a


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
    def test_1darray(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((5,), xp, dtype)
            val = self._compute_val(xp)
            with pytest.raises(ValueError):
                xp.fill_diagonal(a, val=val, wrap=self.wrap)


@testing.parameterize(*testing.product({
    'n': [2, 4, -3, 0],
    'ndim': [2, 3, 1, 0, -2],
}))
@testing.gpu
class TestDiagIndices(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_diag_indices(self, xp):
        return xp.diag_indices(self.n, self.ndim)


@testing.parameterize(*testing.product({
    'n': [-3, 0],
    'ndim': [1, 0, -2],
}))
@testing.gpu
class TestDiagIndicesInvalidValues(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_diag_indices(self, xp):
        return xp.diag_indices(self.n, self.ndim)


@testing.parameterize(*testing.product({
    'shape': [(3, 3), (0, 0), (2, 2, 2)],
}))
@testing.gpu
class TestDiagIndicesFrom(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_diag_indices_from(self, xp):
        arr = testing.shaped_arange(self.shape, xp)
        return xp.diag_indices_from(arr)


@testing.parameterize(*testing.product({
    'shape': [(3, 5), (3, 3, 4), (5,), (0,), (-1,)],
}))
@testing.gpu
class TestDiagIndicesFromRaises(unittest.TestCase):

    def test_non_equal_dims(self):
        for xp in (numpy, cupy):
            arr = testing.shaped_arange(self.shape, xp)
            with pytest.raises(ValueError):
                xp.diag_indices_from(arr)
