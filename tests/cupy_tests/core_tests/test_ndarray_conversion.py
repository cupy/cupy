import unittest

from cupy import testing


@testing.parameterize(
    {'shape': ()},
    {'shape': (1,)},
    {'shape': (1, 1, 1)},
)
class TestNdarrayItem(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_equal()
    def test_item(self, xp, dtype):
        a = xp.full(self.shape, 3, dtype)
        return a.item()


@testing.parameterize(
    {'shape': (0,)},
    {'shape': (2, 3)},
    {'shape': (1, 0, 1)},
)
class TestNdarrayItemRaise(unittest.TestCase):

    @testing.numpy_cupy_raises()
    def test_item(self, xp):
        a = testing.shaped_arange(self.shape, xp, xp.float32)
        a.item()
