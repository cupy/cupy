import unittest

import cupy
from cupy.core import core
from cupy import testing


@testing.parameterize(*testing.product({
    'dtype': [
        cupy.uint32, cupy.int32, cupy.float16, cupy.float32, cupy.float64],
    'shape': [(1,), (2, 3)]
}))
class TestDLPackConversion(unittest.TestCase):

    def setUp(self):
        self.array = cupy.random.rand(*self.shape).astype(self.dtype)

    def test_conversion(self):
        tensor = self.array.toDLPack()
        array = cupy.fromDLPack(tensor)

        testing.assert_array_equal(tensor, array)
