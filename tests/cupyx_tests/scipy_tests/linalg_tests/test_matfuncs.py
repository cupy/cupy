from matfuncs import sinhm
import numpy
import unittest


class TestMatFuncsNan(unittest.TestCase):

    def test_sinhm_values(self):
        # Testing whether it is producing right output or not
        z = numpy.array([
                        [1, 2], [3, 1]])

        r = numpy.array([[1.1752011936438014, 3.626860407847019],
                         [10.017874927409903, 1.1752011936438014]])

        assert numpy.alltrue(sinhm(z) == r)

    def test_values_nan(self):
        # Testing for nan values in matrix
        r = numpy.empty((3, 3,))
        r.fill(numpy.nan)
        self.assertRaises(ValueError, sinhm, r)

    def test_values_inf(self):

        # Testing for nan values in matrix
        z = numpy.zeros([2, 2])
        z[0][0] = numpy.inf
        self.assertRaises(ValueError, sinhm, z)

    def test_shape(self):

        # Testing for matrix shape
        k = numpy.zeros([3, 2])
        self.assertRaises(ValueError, sinhm, k)

    def test_dimension_count(self):
        # Testing whether it is as 2D array or not
        g = numpy.zeros([3, 3, 3])
        self.assertRaises(ValueError, sinhm, g)
