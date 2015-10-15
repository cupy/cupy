import unittest

import numpy

import cupy
from cupy.cuda import curand


class TestGenerateNormal(unittest.TestCase):

    def setUp(self):
        self.generator = curand.createGenerator(
            curand.CURAND_RNG_PSEUDO_DEFAULT)

    def test_invalid_argument_normal_float(self):
        out = cupy.empty((1,), dtype=numpy.float32)
        with self.assertRaises(ValueError):
            curand.generateNormal(
                self.generator, out.data.ptr, 1, 0.0, 1.0)

    def test_invalid_argument_normal_double(self):
        out = cupy.empty((1,), dtype=numpy.float64)
        with self.assertRaises(ValueError):
            curand.generateNormalDouble(
                self.generator, out.data.ptr, 1, 0.0, 1.0)

    def test_invalid_argument_log_normal_float(self):
        out = cupy.empty((1,), dtype=numpy.float32)
        with self.assertRaises(ValueError):
            curand.generateLogNormal(
                self.generator, out.data.ptr, 1, 0.0, 1.0)

    def test_invalid_argument_log_normal_double(self):
        out = cupy.empty((1,), dtype=numpy.float64)
        with self.assertRaises(ValueError):
            curand.generateLogNormalDouble(
                self.generator, out.data.ptr, 1, 0.0, 1.0)
