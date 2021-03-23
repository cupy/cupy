from matfuncs import sinhm
import numpy
import unittest


class TestMatFuncsNan(unittest.TestCase):

    "Testing for nan values in matrix"

    def test_values_nan(self):
        r = numpy.empty((3, 3,))
        r.fill(numpy.nan)
        self.assertRaises(ValueError, sinhm, r)

    "Testing for infinity values in matrix"

    def test_values_inf(self):
        z = numpy.zeros([2, 2])
        z[0][0] = numpy.inf
        self.assertRaises(ValueError, sinhm, z)

    "Testing for matrix shape "

    def test_shape(self):
        k = numpy.zeros([3, 2])
        self.assertRaises(ValueError, sinhm, k)

    "Testing whether it is as 2D array or not"

    def test_dimension_count(self):
        g = numpy.zeros([3, 3, 3])
        self.assertRaises(ValueError, sinhm, g)
