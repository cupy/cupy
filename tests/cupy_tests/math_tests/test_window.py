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
