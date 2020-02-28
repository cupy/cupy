import numpy
import unittest

from cupy import testing


@testing.gpu
class TestPacking(unittest.TestCase):

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def check_packbits(self, data, xp, dtype):
        # Note numpy <= 1.9 raises an Exception when an input array is bool.
        # See https://github.com/numpy/numpy/issues/5377
        a = xp.array(data, dtype=dtype)
        return xp.packbits(a)

    @testing.numpy_cupy_array_equal()
    def check_unpackbits(self, data, xp):
        a = xp.array(data, dtype=xp.uint8)
        return xp.unpackbits(a)

    def test_packbits(self):
        self.check_packbits([0])
        self.check_packbits([1])
        self.check_packbits([0, 1])
        self.check_packbits([1, 0, 1, 1, 0, 1, 1, 1])
        self.check_packbits([1, 0, 1, 1, 0, 1, 1, 1, 1])
        self.check_packbits(numpy.arange(24).reshape((2, 3, 4)) % 2)

    def test_packbits_empty(self):
        # Note packbits of numpy <= 1.11 has a bug against empty arrays.
        # See https://github.com/numpy/numpy/issues/8324
        self.check_packbits([])

    def test_unpackbits(self):
        self.check_unpackbits([])
        self.check_unpackbits([0])
        self.check_unpackbits([1])
        self.check_unpackbits([255])
        self.check_unpackbits([100, 200, 123, 213])
