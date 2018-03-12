import unittest

import cupy
from cupy.core import core
from cupy import testing


class TestGetStridesForNocopyReshape(unittest.TestCase):

    def test_different_size(self):
        a = core.ndarray((2, 3))
        self.assertEqual(core._get_strides_for_nocopy_reshape(a, (1, 5)),
                         [])

    def test_one(self):
        a = core.ndarray((1,), dtype=cupy.int32)
        self.assertEqual(core._get_strides_for_nocopy_reshape(a, (1, 1, 1)),
                         [4, 4, 4])

    def test_normal(self):
        # TODO(nno): write test for normal case
        pass


class TestSize(unittest.TestCase):

    def tearDown(self):
        # Free huge memory for slow test
        cupy.get_default_memory_pool().free_all_blocks()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_size(self, xp, dtype):
        a = xp.ndarray((2, 3), dtype=dtype)
        return xp.size(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_size_axis(self, xp, dtype):
        a = xp.ndarray((2, 3), dtype=dtype)
        return xp.size(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_size_axis_error(self, xp, dtype):
        a = xp.ndarray((2, 3), dtype=dtype)
        return xp.size(a, axis=3)

    @testing.numpy_cupy_equal()
    @testing.slow
    def test_size_huge(self, xp):
        a = xp.ndarray(2 ** 32, 'b')  # 4 GiB
        return xp.size(a)
