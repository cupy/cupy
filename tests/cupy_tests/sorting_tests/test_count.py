import unittest

from cupy import testing


@testing.gpu
class TestCount(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=False)
    def test_count_nonzero(self, xp, dtype):
        m = testing.shaped_random((2, 3), xp, xp.bool_)
        a = testing.shaped_random((2, 3), xp, dtype) * m
        c = xp.count_nonzero(a)
        self.assertIsInstance(c, int)
        return c

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=False)
    def test_count_nonzero_zero_dim(self, xp, dtype):
        a = xp.array(1.0, dtype=dtype)
        c = xp.count_nonzero(a)
        self.assertIsInstance(c, int)
        return c
