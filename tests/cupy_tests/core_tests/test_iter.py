import unittest

from cupy import testing


@testing.gpu
@testing.parameterize(*testing.product(
    {'shape': [(3,), (2, 3, 4), (0,), (0, 2), (3, 0)]},
))
class TestIter(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_list_equal()
    def test_list(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype)
        return list(x)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_len(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype)
        return len(x)


@testing.gpu
class TestIterInvalid(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_iter(self, xp, dtype):
        x = testing.shaped_arange((), xp, dtype)
        iter(x)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_raises()
    def test_len(self, xp, dtype):
        x = testing.shaped_arange((), xp, dtype)
        len(x)
