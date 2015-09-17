import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestArrayBoolOp(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    def test_bool_empty(self, dtype):
        self.assertFalse(bool(cupy.array((), dtype=dtype)))

    def test_bool_scalar_bool(self):
        self.assertTrue(bool(cupy.array(True, dtype=numpy.bool)))
        self.assertFalse(bool(cupy.array(False, dtype=numpy.bool)))

    @testing.for_all_dtypes()
    def test_bool_scalar(self, dtype):
        self.assertTrue(bool(cupy.array(1, dtype=dtype)))
        self.assertFalse(bool(cupy.array(0, dtype=dtype)))

    def test_bool_one_element_bool(self):
        self.assertTrue(bool(cupy.array([True], dtype=numpy.bool)))
        self.assertFalse(bool(cupy.array([False], dtype=numpy.bool)))

    @testing.for_all_dtypes()
    def test_bool_one_element(self, dtype):
        self.assertTrue(bool(cupy.array([1], dtype=dtype)))
        self.assertFalse(bool(cupy.array([0], dtype=dtype)))

    @testing.for_all_dtypes()
    def test_bool_two_elements(self, dtype):
        with self.assertRaises(ValueError):
            bool(cupy.array([1, 2], dtype=dtype))
