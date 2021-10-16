import numpy
import unittest
import pytest
import cupy
from cupy import testing


@testing.gpu
class TestPacking(unittest.TestCase):

    @testing.for_int_dtypes()
    @testing.numpy_cupy_array_equal()
    def check_packbits(self, data, xp, dtype, bitorder='big'):
        # Note numpy <= 1.9 raises an Exception when an input array is bool.
        # See https://github.com/numpy/numpy/issues/5377
        a = xp.array(data, dtype=dtype)
        return xp.packbits(a, bitorder=bitorder)

    @testing.numpy_cupy_array_equal()
    def check_unpackbits(self, data, xp, bitorder='big'):
        a = xp.array(data, dtype=xp.uint8)
        return xp.unpackbits(a, bitorder=bitorder)

    def test_packbits(self):
        self.check_packbits([0])
        self.check_packbits([1])
        self.check_packbits([0, 1])
        self.check_packbits([1, 0, 1, 1, 0, 1, 1, 1])
        self.check_packbits([1, 0, 1, 1, 0, 1, 1, 1, 1])
        self.check_packbits(numpy.arange(24).reshape((2, 3, 4)) % 2)

    def test_packbits_order(self):
        for bo in ['big', 'little']:
            self.check_packbits([0], bitorder=bo)
            self.check_packbits([1], bitorder=bo)
            self.check_packbits([0, 1], bitorder=bo)
            self.check_packbits([1, 0, 1, 1, 0, 1, 1, 1], bitorder=bo)
            self.check_packbits([1, 0, 1, 1, 0, 1, 1, 1, 1], bitorder=bo)
            self.check_packbits(numpy.arange(24).reshape((2, 3, 4)) % 2,
                                bitorder=bo)

    def test_packbits_empty(self):
        # Note packbits of numpy <= 1.11 has a bug against empty arrays.
        # See https://github.com/numpy/numpy/issues/8324
        self.check_packbits([])

    def test_pack_invalid_order(self):
        a = cupy.array([10, 20, 30])
        pytest.raises(ValueError, cupy.packbits, a, bitorder='ascendant')
        pytest.raises(ValueError, cupy.packbits, a, bitorder=10.4)

    def test_pack_invalid_array(self):
        fa = cupy.array([10, 20, 30], dtype=float)
        pytest.raises(TypeError, cupy.packbits, fa)

    def test_unpackbits(self):
        self.check_unpackbits([])
        self.check_unpackbits([0])
        self.check_unpackbits([1])
        self.check_unpackbits([255])
        self.check_unpackbits([100, 200, 123, 213])

    def test_unpack_invalid_array(self):
        a = cupy.array([10, 20, 30])
        pytest.raises(TypeError, cupy.unpackbits, a)
        pytest.raises(TypeError, cupy.unpackbits, a.astype(float))

    def test_pack_unpack_order(self):
        for bo in ['big', 'little']:
            self.check_unpackbits([], bitorder=bo)
            self.check_unpackbits([0], bitorder=bo)
            self.check_unpackbits([1], bitorder=bo)
            self.check_unpackbits([255], bitorder=bo)
            self.check_unpackbits([100, 200, 123, 213], bitorder=bo)

    def test_unpack_invalid_order(self):
        a = cupy.array([10, 20, 30], dtype=cupy.uint8)
        pytest.raises(ValueError, cupy.unpackbits, a, bitorder='r')
        pytest.raises(ValueError, cupy.unpackbits, a, bitorder=10)
