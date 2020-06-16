import numpy
import unittest

import cupy
from cupy import testing


@testing.gpu
class TestMayShareMemory(unittest.TestCase):

    @testing.numpy_cupy_equal()
    def test_different_arrays(self, xp):
        a = xp.array([1, 2, 3])
        b = xp.array([1, 2, 3])
        assert xp.may_share_memory(a, b) is False

    @testing.numpy_cupy_equal()
    def test_same_array(self, xp):
        a = xp.array([1, 2, 3])
        assert xp.may_share_memory(a, a) is True

    @testing.numpy_cupy_equal()
    def test_zero_size(self, xp):
        a = xp.array([])
        assert xp.may_share_memory(a, a) is False

    @testing.numpy_cupy_equal()
    def test_shares_memory(self, xp):
        x = xp.arange(12)
        a = x[0:7]
        b = x[6:12]
        assert xp.may_share_memory(a, b) is True

    @testing.numpy_cupy_equal()
    def test_cover(self, xp):
        x = xp.arange(12)
        a = x[1:10]
        b = x[4:6]
        assert xp.may_share_memory(a, b) is True

    @testing.numpy_cupy_equal()
    def test_away(self, xp):
        x = xp.arange(12)
        a = x[1:6]
        b = x[8:11]
        assert xp.may_share_memory(a, b) is False

    @testing.numpy_cupy_equal()
    def test_touch_edge_true(self, xp):
        x = xp.arange(12)
        a = x[1:10]
        b = x[7:10]
        assert xp.may_share_memory(a, b) is True

    @testing.numpy_cupy_equal()
    def test_touch_edge_false(self, xp):
        x = xp.arange(12)
        a = x[1:7]
        b = x[7:10]
        assert xp.may_share_memory(a, b) is False

    def _get_slices(self, size):
        slices = []
        for start in range(0, size + 1):
            for end in range(start, size + 1):
                for step in range(-2, 2):
                    if step != 0:
                        slices.append(slice(start, end, step))
        return slices

    def test_combination(self):
        size = 4
        slices = self._get_slices(size)
        memory_np = numpy.empty(size * size)
        memory_cp = cupy.empty(size * size)

        arrays = []

        array_1d_np = memory_np[5:5+size]
        array_1d_cp = memory_cp[5:5+size]
        for s in slices:
            arrays.append((array_1d_np[s], array_1d_cp[s], s))

        array_2d_np = memory_np.reshape(size, size)
        array_2d_cp = memory_cp.reshape(size, size)
        for s1 in slices:
            for s2 in slices:
                arrays.append((
                    array_2d_np[s1, s2], array_2d_cp[s1, s2], (s1, s2)))

        for array1_np, array1_cp, sl1 in arrays:
            for array2_np, array2_cp, sl2 in arrays:
                ret_np = numpy.may_share_memory(array1_np, array2_np)
                ret_cp = cupy.may_share_memory(array1_cp, array2_cp)
                assert ret_np == ret_cp, \
                    'Failed in case of {} and {}'.format(sl1, sl2)


@testing.gpu
class TestSharesMemory(unittest.TestCase):

    def test_different_arrays(self):
        for xp in (numpy, cupy):
            a = xp.array([1, 2, 3])
            b = xp.array([1, 2, 3])
            assert xp.shares_memory(a, b) is False

    def test_same_array(self):
        for xp in (numpy, cupy):
            a = xp.array([1, 2, 3])
            assert xp.shares_memory(a, a) is True

    def test_zero_size_array(self):
        for xp in (numpy, cupy):
            a = xp.array([])
            assert xp.shares_memory(a, a) is False

    def test_contiguous_arrays(self):
        for xp in (numpy, cupy):
            x = xp.arange(12)
            # shares memory
            assert xp.shares_memory(x[0:7], x[6:12]) is True
            # covers
            assert xp.shares_memory(x[1:10], x[4:6]) is True
            assert xp.shares_memory(x[4:6], x[1:10]) is True
            # detached
            assert xp.shares_memory(x[1:6], x[8:11]) is False
            # touch
            assert xp.shares_memory(x[1:10], x[7:10]) is True
            assert xp.shares_memory(x[1:7], x[7:10]) is False

    def test_non_contiguous_case(self):
        for xp in (numpy, cupy):
            x = xp.arange(100)
            assert xp.shares_memory(x, x[1::4]) is True
            assert xp.shares_memory(x[0::2], x[1::4]) is False
            assert xp.shares_memory(x[0::9], x[1::11]) is True

    def test_multi_dimension_case(self):
        for xp in (numpy, cupy):
            x = xp.arange(100).reshape(10, 10)
            assert xp.shares_memory(x[0::2], x[1::3]) is True
            assert xp.shares_memory(x[0::2], x[1::4]) is False
            assert xp.shares_memory(x[0::2], x[::, 1::2]) is True

    def test_complex_type_case(self):
        for xp in (numpy, cupy):
            x = testing.shaped_random((2, 3, 4), xp, numpy.complex128)
            assert xp.shares_memory(x, x.imag) is True
