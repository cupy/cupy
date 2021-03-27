from cupyx.scipy.linalg.matfuncs import sinhm
import cupy
import unittest


class TestMatFuncsNan(unittest.TestCase):

    def test_sinhm_values(self):
        # Testing whether it is producing right output or not
        z = cupy.zeros((2, 2))
        z[0][0] = 1
        z[0][1] = 2
        z[1][0] = 3
        z[1][1] = 1

        r = cupy.zeros((2, 2))

        r[0][0] = 1.1752011936438014
        r[0][1] = 3.626860407847019
        r[1][0] = 10.017874927409903
        r[1][1] = 1.1752011936438014

        z = sinhm(z)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                assert r[i][j] == z[i][j]

    def test_values_nan(self):
        # Testing for nan values in matrix
        r = cupy.zeros((3, 3))
        r.fill(cupy.nan)
        self.assertRaises(ValueError, sinhm, r)

    def test_values_inf(self):

        # Testing for nan values in matrix
        z = cupy.zeros((2, 2))
        z[0][0] = cupy.inf
        self.assertRaises(ValueError, sinhm, z)

    def test_shape(self):

        # Testing for matrix shape
        k = cupy.zeros((3, 2))
        self.assertRaises(ValueError, sinhm, k)

    def test_dimension_count(self):
        # Testing whether it is as 2D array or not
        g = cupy.zeros((3, 3, 3))
        self.assertRaises(ValueError, sinhm, g)
