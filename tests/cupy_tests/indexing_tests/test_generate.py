import unittest

from cupy import testing


@testing.gpu
class TestIX_(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_array_list_equal()
    def test_ix_list(self, xp):
        return xp.ix_([0, 1], [2, 4])

    @testing.numpy_cupy_array_list_equal()
    def test_ix_ndarray(self, xp):
        return xp.ix_(xp.array([0, 1]), xp.array([2, 4]))

    @testing.numpy_cupy_array_list_equal()
    def test_ix_ndarray_int(self, xp):
        return xp.ix_(xp.array([0, 1]), xp.array([2, 4]))

    @testing.numpy_cupy_array_list_equal()
    def test_ix_empty_ndarray(self, xp):
        return xp.ix_(xp.array([]))

    @testing.numpy_cupy_array_list_equal()
    def test_ix_bool_ndarray(self, xp):
        return xp.ix_(xp.array([True, False] * 2))
