import unittest

from cupy import testing


@testing.parameterize(*testing.product({
    'shape': [(3, 3), (2, 2, 2), (3, 5), (5, 3)],
    'val': [1, 0],
    'wrap': [True, False],
}))
@testing.gpu
class TestInsert(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_fill_diagonal(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        xp.fill_diagonal(a, val=self.val, wrap=self.wrap)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_1darray(self, xp, dtype):
        a = testing.shaped_arange(5, xp, dtype)
        xp.fill_diagonal(a, val=self.val, wrap=self.wrap)
