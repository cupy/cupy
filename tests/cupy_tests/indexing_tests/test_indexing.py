import unittest

import numpy
import pytest

import cupy
from cupy import testing


@testing.gpu
class TestIndexing(unittest.TestCase):

    @testing.numpy_cupy_array_equal()
    def test_take_by_scalar(self, xp):
        a = testing.shaped_arange((2, 4, 3), xp)
        return a.take(2, axis=1)

    @testing.numpy_cupy_array_equal()
    def test_external_take_by_scalar(self, xp):
        a = testing.shaped_arange((2, 4, 3), xp)
        return xp.take(a, 2, axis=1)

    @testing.numpy_cupy_array_equal()
    def test_take_by_array(self, xp):
        a = testing.shaped_arange((2, 4, 3), xp)
        b = xp.array([[1, 3], [2, 0]])
        return a.take(b, axis=1)

    @testing.numpy_cupy_array_equal()
    def test_take_no_axis(self, xp):
        a = testing.shaped_arange((2, 3, 4), xp)
        b = xp.array([[10, 5], [3, 20]])
        return a.take(b)

    # see cupy#3017
    # mark slow as NumPy could go OOM on the Windows CI
    @testing.slow
    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_take_index_range_overflow(self, xp, dtype):
        # Skip for too large dimensions
        if numpy.dtype(dtype) in (numpy.int64, numpy.uint64):
            pytest.skip()
        # Skip because NumPy actually allocates a contiguous array in the
        # `take` below to require much time.
        if dtype in (numpy.int32, numpy.uint32):
            pytest.skip()
        iinfo = numpy.iinfo(dtype)
        a = xp.broadcast_to(xp.ones(1), (iinfo.max + 1,))
        b = xp.array([0], dtype=dtype)
        return a.take(b)

    @testing.numpy_cupy_array_equal()
    def test_take_along_axis(self, xp):
        a = testing.shaped_random((2, 4, 3), xp, dtype='float32')
        b = testing.shaped_random((2, 6, 3), xp, dtype='int64', scale=4)
        return xp.take_along_axis(a, b, axis=-2)

    @testing.numpy_cupy_array_equal()
    def test_take_along_axis_none_axis(self, xp):
        a = testing.shaped_random((2, 4, 3), xp, dtype='float32')
        b = testing.shaped_random((30,), xp, dtype='int64', scale=24)
        return xp.take_along_axis(a, b, axis=None)

    @testing.numpy_cupy_array_equal()
    def test_compress(self, xp):
        a = testing.shaped_arange((3, 4, 5), xp)
        b = xp.array([True, False, True])
        return xp.compress(b, a, axis=1)

    @testing.numpy_cupy_array_equal()
    def test_compress_no_axis(self, xp):
        a = testing.shaped_arange((3, 4, 5), xp)
        b = xp.array([True, False, True])
        return xp.compress(b, a)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_compress_no_bool(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp)
        b = testing.shaped_arange((3,), xp, dtype)
        return xp.compress(b, a, axis=1)

    @testing.numpy_cupy_array_equal()
    def test_compress_empty_1dim(self, xp):
        a = testing.shaped_arange((3, 4, 5), xp)
        b = xp.array([])
        return xp.compress(b, a, axis=1)

    @testing.numpy_cupy_array_equal()
    def test_compress_empty_1dim_no_axis(self, xp):
        a = testing.shaped_arange((3, 4, 5), xp)
        b = xp.array([])
        return xp.compress(b, a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return a.diagonal(1, 2, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_external_diagonal(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return xp.diagonal(a, 1, 2, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal_negative1(self, xp, dtype):
        a = testing.shaped_arange((3, 4, 5), xp, dtype)
        return a.diagonal(-1, 2, 0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal_negative2(self, xp, dtype):
        a = testing.shaped_arange((3, 3, 3), xp, dtype)
        return a.diagonal(0, -1, -2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal_negative3(self, xp, dtype):
        a = testing.shaped_arange((3, 3, 3), xp, dtype)
        return a.diagonal(0, -1, 1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal_negative4(self, xp, dtype):
        a = testing.shaped_arange((3, 3, 3), xp, dtype)
        return a.diagonal(0, -3, -1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_diagonal_negative5(self, xp, dtype):
        a = testing.shaped_arange((3, 3, 3), xp, dtype)
        return a.diagonal(0, -1, -3)

    def test_diagonal_invalid1(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((3, 3, 3), xp)
            with pytest.raises(IndexError):
                a.diagonal(0, 1, 3)

    def test_diagonal_invalid2(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((3, 3, 3), xp)
            with pytest.raises(IndexError):
                a.diagonal(0, 2, -4)

    @testing.numpy_cupy_array_equal()
    def test_extract(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        b = xp.array([[True, False, True],
                      [False, True, False],
                      [True, False, True]])
        return xp.extract(b, a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_extract_no_bool(self, xp, dtype):
        a = testing.shaped_arange((3, 3), xp)
        b = xp.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=dtype)
        return xp.extract(b, a)

    @testing.numpy_cupy_array_equal()
    def test_extract_shape_mismatch(self, xp):
        a = testing.shaped_arange((2, 3), xp)
        b = xp.array([[True, False],
                      [True, False],
                      [True, False]])
        return xp.extract(b, a)

    @testing.numpy_cupy_array_equal()
    def test_extract_size_mismatch(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        b = xp.array([[True, False, True],
                      [False, True, False]])
        return xp.extract(b, a)

    @testing.numpy_cupy_array_equal()
    def test_extract_size_mismatch2(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        b = xp.array([[True, False, True, False],
                      [False, True, False, True]])
        return xp.extract(b, a)

    @testing.numpy_cupy_array_equal()
    def test_extract_empty_1dim(self, xp):
        a = testing.shaped_arange((3, 3), xp)
        b = xp.array([])
        return xp.extract(b, a)


@testing.gpu
class TestChoose(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_choose(self, xp, dtype):
        a = xp.array([0, 2, 1, 2])
        c = testing.shaped_arange((3, 4), xp, dtype)
        return a.choose(c)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_choose_broadcast(self, xp, dtype):
        a = xp.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        c = xp.array([-10, 10], dtype=dtype)
        return a.choose(c)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_choose_broadcast2(self, xp, dtype):
        a = xp.array([0, 1])
        c = testing.shaped_arange((3, 5, 2), xp, dtype)
        return a.choose(c)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_choose_wrap(self, xp, dtype):
        a = xp.array([0, 3, -1, 5])
        c = testing.shaped_arange((3, 4), xp, dtype)
        return a.choose(c, mode='wrap')

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_choose_clip(self, xp, dtype):
        a = xp.array([0, 3, -1, 5])
        c = testing.shaped_arange((3, 4), xp, dtype)
        return a.choose(c, mode='clip')

    @testing.with_requires('numpy>=1.19')
    def test_unknown_clip(self):
        for xp in (numpy, cupy):
            a = xp.array([0, 3, -1, 5])
            c = testing.shaped_arange((3, 4), xp, numpy.float32)
            with pytest.raises(ValueError):
                a.choose(c, mode='unknow')

    def test_raise(self):
        a = cupy.array([2])
        c = cupy.array([[0, 1]])
        with self.assertRaises(ValueError):
            a.choose(c)

    @testing.for_all_dtypes()
    def test_choose_broadcast_fail(self, dtype):
        for xp in (numpy, cupy):
            a = xp.array([0, 1])
            c = testing.shaped_arange((3, 5, 4), xp, dtype)
            with pytest.raises(ValueError):
                return a.choose(c)


@testing.gpu
class TestSelect(unittest.TestCase):

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_select(self, xp, dtype):
        a = xp.arange(10, dtype=dtype)
        condlist = [a > 3, a < 5]
        choicelist = [a, a**2]
        return xp.select(condlist, choicelist)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_array_almost_equal()
    def test_select_complex(self, xp, dtype):
        a = xp.arange(10, dtype=dtype)
        condlist = [a > 3, a < 5]
        choicelist = [a, a**2]
        return xp.select(condlist, choicelist)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_select_default(self, xp, dtype):
        a = xp.arange(10, dtype=dtype)
        condlist = [a > 3, a < 5]
        choicelist = [a, a**2]
        default = 3
        return xp.select(condlist, choicelist, default)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_array_almost_equal()
    def test_select_default_complex(self, xp, dtype):
        a = xp.arange(10, dtype=dtype)
        condlist = [a > 3, a < 5]
        choicelist = [a, a**2]
        default = 3
        return xp.select(condlist, choicelist, default)

    @testing.for_all_dtypes(no_bool=True, no_complex=True)
    @testing.numpy_cupy_array_equal()
    def test_select_odd_shaped_broadcastable(self, xp, dtype):
        a = xp.arange(10, dtype=dtype)
        b = xp.arange(30, dtype=dtype).reshape(3, 10)
        condlist = [a < 3, b > 8]
        choicelist = [a, b]
        return xp.select(condlist, choicelist)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_select_odd_shaped_broadcastable_complex(self, xp, dtype):
        a = xp.arange(10, dtype=dtype)
        b = xp.arange(20, dtype=dtype).reshape(2, 10)
        condlist = [a < 3, b > 8]
        choicelist = [a, b**2]
        return xp.select(condlist, choicelist)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_select_1D_choicelist(self, xp, dtype):
        a = xp.array(1)
        b = xp.array(3)
        condlist = [a < 3, b > 8]
        choicelist = [a, b]
        return xp.select(condlist, choicelist)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_select_choicelist_condlist_broadcast(self, xp, dtype):
        a = xp.arange(10, dtype=dtype)
        b = xp.arange(20, dtype=dtype).reshape(2, 10)
        condlist = [a < 4, b > 8]
        choicelist = [xp.repeat(a, 2).reshape(2, 10), b]
        return xp.select(condlist, choicelist)

    @testing.for_all_dtypes(no_bool=True)
    def test_select_length_error(self, dtype):
        a = cupy.arange(10, dtype=dtype)
        condlist = [a > 3]
        choicelist = [a, a**2]
        with pytest.raises(ValueError):
            cupy.select(condlist, choicelist)

    @testing.for_all_dtypes(no_bool=True)
    def test_select_type_error_condlist(self, dtype):
        a = cupy.arange(10, dtype=dtype)
        condlist = [[3] * 10, [2] * 10]
        choicelist = [a, a**2]
        with pytest.raises(AttributeError):
            cupy.select(condlist, choicelist)

    @testing.for_all_dtypes(no_bool=True)
    def test_select_type_error_choicelist(self, dtype):
        a, b = list(range(10)), list(range(-10, 0))
        condlist = [0] * 10
        choicelist = [a, b]
        with pytest.raises(ValueError):
            cupy.select(condlist, choicelist)

    def test_select_empty_lists(self):
        condlist = []
        choicelist = []
        with pytest.raises(ValueError):
            cupy.select(condlist, choicelist)

    @testing.for_all_dtypes(no_bool=True)
    def test_select_odd_shaped_non_broadcastable(self, dtype):
        a = cupy.arange(10, dtype=dtype)
        b = cupy.arange(20, dtype=dtype)
        condlist = [a < 3, b > 8]
        choicelist = [a, b]
        with pytest.raises(ValueError):
            cupy.select(condlist, choicelist)

    @testing.for_all_dtypes(no_bool=True)
    def test_select_default_scalar(self, dtype):
        a = cupy.arange(10)
        b = cupy.arange(20)
        condlist = [a < 3, b > 8]
        choicelist = [a, b]
        with pytest.raises(TypeError):
            cupy.select(condlist, choicelist, [dtype(2)])
