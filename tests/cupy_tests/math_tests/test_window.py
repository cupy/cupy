import unittest

from cupy import testing


@testing.parameterize(
    *testing.product({
        'm': [0, 1, -1, 1024],
        'name': ['blackman', 'hamming', 'hanning'],
    })
)
class TestWindow(unittest.TestCase):

    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_window(self, xp):
        return getattr(xp, self.name)(self.m)


@testing.parameterize(
    *testing.product({
        'm': [0, 1, -1, 1024],
        'beta': [0, 5, 6, 8.6],
        'name': ['kaiser'],
    })
)
class TestKaiser(unittest.TestCase):

    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_kaiser(self, xp):
        return getattr(xp, self.name)(self.m, self.beta)
