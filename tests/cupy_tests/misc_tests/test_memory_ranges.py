import unittest

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

    @testing.numpy_cupy_equal()
    def test_combination(self, xp):
        size = 4

        slices = []
        for start in range(0, size + 1):
            for end in range(start, size + 1):
                for step in range(-2, 2):
                    if step != 0:
                        slices.append(slice(start, end, step))

        memory = xp.empty(size * size)
        arrays = []

        array_1d = memory[5:5+size]
        for s in slices:
            arrays.append(array_1d[s])

        array_2d = memory.reshape(size, size)
        for s1 in slices:
            for s2 in slices:
                arrays.append(array_2d[s1, s2])

        result = []

        for array1 in arrays:
            for array2 in arrays:
                ret = xp.may_share_memory(array1, array2)
                result.append(ret)

        return result
