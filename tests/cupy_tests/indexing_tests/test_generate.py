import unittest

import cupy
from cupy.indexing import generate
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

    @testing.numpy_cupy_raises()
    def test_indices_list3(self, xp):
        return xp.indices((1, 2, 3, 4), dtype=xp.bool_)


@testing.gpu
class TestIX_(unittest.TestCase):

    @testing.numpy_cupy_array_list_equal()
    def test_ix_list(self, xp):
        return xp.ix_([0, 1], [2, 4])

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_ix_ndarray(self, xp, dtype):
        return xp.ix_(xp.array([0, 1], dtype), xp.array([2, 3], dtype))

    @testing.numpy_cupy_array_list_equal()
    def test_ix_empty_ndarray(self, xp):
        return xp.ix_(xp.array([]))

    @testing.numpy_cupy_array_list_equal()
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
            cupy.indexing.generate.AxisConcatenator.__init__()

    def test_len(self):
        a = generate.AxisConcatenator()
        self.assertEqual(len(a), 0)


@testing.gpu
class TestUnravelIndex(unittest.TestCase):

    @testing.for_orders(['C', 'F', None])
    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test(self, xp, order, dtype):
        a = testing.shaped_arange((4, 3, 2), xp, dtype)
        a = xp.minimum(a, 6 * 4 - 1)
        return xp.unravel_index(a, (6, 4), order=order)

    @testing.for_int_dtypes()
    @testing.numpy_cupy_raises(accept_error=TypeError)
    def test_invalid_order(self, xp, dtype):
        a = testing.shaped_arange((4, 3, 2), xp, dtype)
        xp.unravel_index(a, (6, 4), order='V')

    @testing.for_orders(['C', 'F', None])
    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_raises(accept_error=ValueError)
    def test_invalid_index(self, xp, order, dtype):
        a = testing.shaped_arange((4, 3, 2), xp, dtype)
        xp.unravel_index(a, (6, 4), order=order)

    @testing.for_orders(['C', 'F', None])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_raises(accept_error=TypeError)
    def test_invalid_dtype(self, xp, order, dtype):
        a = testing.shaped_arange((4, 3, 2), xp, dtype)
        xp.unravel_index(a, (6, 4), order=order)
