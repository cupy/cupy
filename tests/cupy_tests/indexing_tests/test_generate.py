import unittest

import numpy
import pytest

import cupy
from cupy.cuda import driver
from cupy.cuda import runtime
from cupy._indexing import generate
from cupy import testing


@testing.gpu
class TestIndices(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_indices_list0(self, xp, dtype):
        return xp.indices((0,), dtype)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_indices_list1(self, xp, dtype):
        return xp.indices((1, 2), dtype)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_indices_list2(self, xp, dtype):
        return xp.indices((1, 2, 3, 4), dtype)

    def test_indices_list3(self):
        for xp in (numpy, cupy):
            with pytest.raises(ValueError):
                xp.indices((1, 2, 3, 4), dtype=xp.bool_)


@testing.gpu
class TestIX_(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_ix_list(self, xp):
        return xp.ix_([0, 1], [2, 4])

    @pytest.mark.xfail(runtime.is_hip and driver.get_build_version() < 402,
                       reason='HIP may have a bug')
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_ix_ndarray(self, xp, dtype):
        return xp.ix_(xp.array([0, 1], dtype), xp.array([2, 3], dtype))

    @testing.numpy_cupy_array_equal(type_check=False)
    def test_ix_empty_ndarray(self, xp):
        return xp.ix_(xp.array([]))

    @testing.numpy_cupy_array_equal()
    def test_ix_bool_ndarray(self, xp):
        return xp.ix_(xp.array([True, False] * 2))


@testing.gpu
class TestR_(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_r_1(self, xp, dtype):
        a = testing.shaped_arange((3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 4), xp, dtype)
        return xp.r_[a, b]

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_r_8(self, xp, dtype):
        a = testing.shaped_arange((3, 4), xp, dtype)
        b = testing.shaped_reverse_arange((2, 4), xp, dtype)
        c = testing.shaped_reverse_arange((1, 4), xp, dtype)
        return xp.r_[a, b, c]

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_r_2(self, xp, dtype):
        a = xp.array([1, 2, 3], dtype)
        return xp.r_[a, 0, 0, a]

    def test_r_3(self):
        with self.assertRaises(NotImplementedError):
            cupy.r_[-1:1:6j, [0] * 3, 5, 6]

    @testing.for_all_dtypes()
    def test_r_4(self, dtype):
        a = testing.shaped_arange((1, 3), cupy, dtype)
        with self.assertRaises(NotImplementedError):
            cupy.r_['-1', a, a]

    def test_r_5(self):
        with self.assertRaises(NotImplementedError):
            cupy.r_['0,2', [1, 2, 3], [4, 5, 6]]

    def test_r_6(self):
        with self.assertRaises(NotImplementedError):
            cupy.r_['0,2,0', [1, 2, 3], [4, 5, 6]]

    def test_r_7(self):
        with self.assertRaises(NotImplementedError):
            cupy.r_['r', [1, 2, 3], [4, 5, 6]]

    @testing.for_all_dtypes()
    def test_r_9(self, dtype):
        a = testing.shaped_arange((3, 4), cupy, dtype)
        b = testing.shaped_reverse_arange((2, 5), cupy, dtype)
        with self.assertRaises(ValueError):
            cupy.r_[a, b]


@testing.gpu
class TestC_(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_c_1(self, xp, dtype):
        a = testing.shaped_arange((4, 2), xp, dtype)
        b = testing.shaped_reverse_arange((4, 3), xp, dtype)
        return xp.c_[a, b]

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_c_2(self, xp, dtype):
        a = testing.shaped_arange((4, 2), xp, dtype)
        b = testing.shaped_reverse_arange((4, 3), xp, dtype)
        c = testing.shaped_reverse_arange((4, 1), xp, dtype)
        return xp.c_[a, b, c]

    @testing.for_all_dtypes()
    def test_c_3(self, dtype):
        a = testing.shaped_arange((3, 4), cupy, dtype)
        b = testing.shaped_reverse_arange((2, 5), cupy, dtype)
        with self.assertRaises(ValueError):
            cupy.c_[a, b]


@testing.gpu
class TestAxisConcatenator(unittest.TestCase):

    def test_AxisConcatenator_init1(self):
        with self.assertRaises(TypeError):
            generate.AxisConcatenator.__init__()

    def test_len(self):
        a = generate.AxisConcatenator()
        assert len(a) == 0


@testing.gpu
class TestUnravelIndex(unittest.TestCase):

    @testing.for_orders(['C', 'F', None])
    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal(type_check=False)
    def test(self, xp, order, dtype):
        a = testing.shaped_arange((4, 3, 2), xp, dtype)
        a = xp.minimum(a, 6 * 4 - 1)
        return xp.unravel_index(a, (6, 4), order=order)

    @testing.with_requires('numpy>=1.19')
    @testing.for_int_dtypes()
    def test_invalid_order(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((4, 3, 2), xp, dtype)
            with pytest.raises(ValueError):
                xp.unravel_index(a, (6, 4), order='V')

    @testing.for_orders(['C', 'F', None])
    @testing.for_int_dtypes(no_bool=True)
    def test_invalid_index(self, order, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((4, 3, 2), xp, dtype)
            with pytest.raises(ValueError):
                xp.unravel_index(a, (6, 4), order=order)

    @testing.for_orders(['C', 'F', None])
    @testing.for_float_dtypes()
    def test_invalid_dtype(self, order, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((4, 3, 2), xp, dtype)
            with pytest.raises(TypeError):
                xp.unravel_index(a, (6, 4), order=order)


@testing.gpu
class TestRavelMultiIndex(unittest.TestCase):

    @testing.for_orders(['C', 'F', None])
    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_basic(self, xp, order, dtype):
        dims = (8, 4)
        a = [xp.ones(5, dtype=dtype)] * len(dims)
        return xp.ravel_multi_index(a, dims, order=order)

    @testing.for_orders(['C', 'F', None])
    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_multi_index_broadcasting(self, xp, order, dtype):
        dims = (3, 5)
        x, y = xp.meshgrid(*[xp.arange(s, dtype=dtype) for s in dims],
                           sparse=True)
        return xp.ravel_multi_index((x, y), dims, order=order)

    @testing.for_orders(['C', 'F', None])
    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_basic_nd_coords(self, xp, order, dtype):
        dims = (8, 4)
        a = [xp.ones((3, 3, 3), dtype=dtype)] * len(dims)
        return xp.ravel_multi_index(a, dims, order=order)

    @testing.for_orders(['C', 'F', None])
    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_basic_clip(self, xp, order, dtype):
        dims = (8, 4, 2)
        a = [xp.arange(max(dims), dtype=dtype)] * len(dims)
        return xp.ravel_multi_index(a, dims, order=order, mode='clip')

    @testing.for_orders(['C', 'F', None])
    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_basic_wrap(self, xp, order, dtype):
        dims = (8, 4, 2)
        a = [xp.arange(max(dims), dtype=dtype)] * len(dims)
        return xp.ravel_multi_index(a, dims, order=order, mode='wrap')

    @testing.for_orders(['C', 'F', None])
    @testing.for_int_dtypes(no_bool=True)
    def test_basic_raise(self, order, dtype):
        for xp in (numpy, cupy):
            dims = (8, 4, 2)
            a = [xp.arange(max(dims), dtype=dtype)] * len(dims)
            with pytest.raises(ValueError):
                return xp.ravel_multi_index(a, dims, order=order, mode='raise')

    @testing.for_int_dtypes()
    def test_invalid_float_dims(self, dtype):
        for xp in (numpy, cupy):
            a = xp.ones((3, 5), dtype=dtype)
            with pytest.raises(TypeError):
                xp.ravel_multi_index(a, (2., 4, 8.))

    @testing.for_float_dtypes()
    def test_invalid_multi_index_dtype(self, dtype):
        for xp in (numpy, cupy):
            a = xp.ones((3, 5), dtype=dtype)
            with pytest.raises(TypeError):
                xp.ravel_multi_index(a, (2, 4, 8))

    @testing.for_orders(['C', 'F', None])
    @testing.for_int_dtypes(no_bool=True)
    def test_invalid_multi_index_shape(self, order, dtype):
        for xp in (numpy, cupy):
            # a.shape[0] != len(dims)
            dims = (2, 4)
            a = xp.ones((len(dims) + 1, 5), dtype=dtype)
            with pytest.raises(ValueError):
                xp.ravel_multi_index(a, dims, order=order)

    @testing.with_requires('numpy>=1.19')
    @testing.for_int_dtypes(no_bool=True)
    def test_invalid_order(self, dtype):
        for xp in (numpy, cupy):
            dims = (8, 4)
            a = tuple([xp.arange(min(dims), dtype=dtype) for d in dims])
            with pytest.raises(ValueError):
                xp.ravel_multi_index(a, dims, order='V')

    @testing.for_orders(['C', 'F', None])
    @testing.for_int_dtypes(no_bool=True)
    def test_dims_overflow(self, order, dtype):
        for xp in (numpy, cupy):
            dims = (8, 4)
            a = tuple([xp.arange(min(dims), dtype=dtype) for d in dims])
            with pytest.raises(ValueError):
                xp.ravel_multi_index(
                    a, (xp.iinfo(xp.int64).max, 4), order=order)

    @testing.with_requires('numpy>=1.19')
    @testing.for_int_dtypes(no_bool=True)
    def test_invalid_mode(self, dtype):
        for xp in (numpy, cupy):
            dims = (8, 4)
            a = tuple([xp.arange(min(dims), dtype=dtype) for d in dims])
            with pytest.raises(ValueError):
                xp.ravel_multi_index(a, dims, mode='invalid')


class TestMaskIndices:

    @testing.numpy_cupy_array_equal()
    def test_mask_indices(self, xp):
        # arr is a square matrix with 50% density
        multiplier = testing.shaped_random((10, 10), xp=xp, dtype=xp.bool_)
        arr = testing.shaped_random((10, 10), xp=xp) * multiplier
        return xp.mask_indices(10, lambda n, k=None: arr)

    @testing.numpy_cupy_array_equal()
    def test_mask_indices_k(self, xp):
        return xp.mask_indices(10, xp.triu, k=1)

    @testing.numpy_cupy_array_equal()
    def test_empty(self, xp):
        return xp.mask_indices(0, xp.triu)


class TestTrilIndices:

    @testing.numpy_cupy_array_equal()
    def test_tril_indices_1(self, xp):
        return xp.tril_indices(n=29, k=0)

    @testing.numpy_cupy_array_equal()
    def test_tril_indices_2(self, xp):
        return xp.tril_indices(n=11, k=4, m=4)

    @testing.numpy_cupy_array_equal()
    def test_tril_indices_3(self, xp):
        return xp.tril_indices(n=4, k=4, m=3)

    @testing.for_all_dtypes()
    def test_tril_indices(self, dtype):
        for xp in (numpy, cupy):
            arr = testing.shaped_random((10, 10), xp=xp, dtype=dtype)
            if xp is numpy:
                error = ValueError
            else:
                error = TypeError
            with pytest.raises(error):
                xp.tril_indices(arr, k=0)


class TestTrilIndicesForm:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_tril_indices_from_1(self, xp, dtype):
        arr = testing.shaped_random((10, 10), xp=xp, dtype=dtype)
        return xp.tril_indices_from(arr, k=4)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_tril_indices_from_2(self, xp, dtype):
        arr = testing.shaped_random((10, 20), xp=xp, dtype=dtype)
        return xp.tril_indices_from(arr, k=13)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_tril_indices_from_3(self, xp, dtype):
        arr = testing.shaped_random((4, 6), xp=xp, dtype=dtype)
        return xp.tril_indices_from(arr)

    @testing.for_all_dtypes()
    def test_tril_indices_from_4(self, dtype):
        for xp in (numpy, cupy):
            with pytest.raises(AttributeError):
                xp.tril_indices_from(4, k=1)


class TestTriuIndices:

    @testing.numpy_cupy_array_equal()
    def test_triu_indices_1(self, xp):
        return xp.triu_indices(n=10, k=0)

    @testing.numpy_cupy_array_equal()
    def test_triu_indices_2(self, xp):
        return xp.triu_indices(n=23, k=3, m=4)

    @testing.numpy_cupy_array_equal()
    def test_triu_indices_3(self, xp):
        return xp.triu_indices(n=4, k=4, m=4)

    @testing.for_all_dtypes()
    def test_triu_indices_4(self, dtype):
        for xp in (numpy, cupy):
            arr = testing.shaped_random((10, 10), xp=xp, dtype=dtype)
            if xp is numpy:
                error = ValueError
            else:
                error = TypeError
            with pytest.raises(error):
                xp.triu_indices(arr, k=0)


class TestTriuIndicesFrom:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_triu_indices_from_1(self, xp, dtype):
        arr = testing.shaped_random((20, 20), xp=xp, dtype=dtype)
        return xp.triu_indices_from(arr, k=11)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_triu_indices_from_2(self, xp, dtype):
        arr = testing.shaped_random((20, 5), xp=xp, dtype=dtype)
        return xp.triu_indices_from(arr, k=32)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_triu_indices_from_3(self, xp, dtype):
        arr = testing.shaped_random((4, 6), xp=xp, dtype=dtype)
        return xp.triu_indices_from(arr)

    @testing.for_all_dtypes()
    def test_triu_indices_from_4(self, dtype):
        for xp in (numpy, cupy):
            with pytest.raises(AttributeError):
                xp.triu_indices_from(4, k=1)
