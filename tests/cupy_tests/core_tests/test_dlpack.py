import unittest

import cupy
from cupy import testing


@testing.parameterize(*testing.product({
    'dtype': [
        cupy.uint8, cupy.uint16, cupy.uint32, cupy.uint64,
        cupy.int8, cupy.int16, cupy.int32, cupy.int64,
        cupy.float16, cupy.float32, cupy.float64],
}))
class TestDLPackConversion(unittest.TestCase):

    def setUp(self):
        if cupy.issubdtype(self.dtype, cupy.unsignedinteger):
            self.array = cupy.random.randint(
                0, 10, size=(2, 3)).astype(self.dtype)
        elif cupy.issubdtype(self.dtype, cupy.integer):
            self.array = cupy.random.randint(
                -10, 10, size=(2, 3)).astype(self.dtype)
        elif cupy.issubdtype(self.dtype, cupy.floating):
            self.array = cupy.random.rand(
                2, 3).astype(self.dtype)

    def test_conversion(self):
        tensor = self.array.toDlpack()
        array = cupy.fromDlpack(tensor)
        testing.assert_array_equal(self.array, array)
