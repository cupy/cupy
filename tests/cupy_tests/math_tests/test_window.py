import unittest

from cupy import testing


@testing.gpu
class TestWindow(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_0(self, name, xp):
        a = 0
        return getattr(xp, name)(a)

    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_1(self, name, xp):
        a = 1
        return getattr(xp, name)(a)

    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_negative(self, name, xp):
        a = -1
        return getattr(xp, name)(a)

    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_large(self, name, xp):
        a = 1024
        return getattr(xp, name)(a)

    def check_all(self, name):
        self.check_0(name)
        self.check_1(name)
        self.check_negative(name)
        self.check_large(name)

    def test_blackman(self):
        self.check_all('blackman')

    def test_hamming(self):
        self.check_all('hamming')

    def test_hanning(self):
        self.check_all('hanning')
