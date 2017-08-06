import unittest

import numpy

from cupy import testing


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'format': ['csr', 'csc', 'coo'],
    'm': [3],
    'n': [None, 3],
}))
@testing.with_requires('scipy')
class TestEye(unittest.TestCase):

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_eye(self, xp, sp):
        x = sp.eye(self.m, n=self.n, dtype=self.dtype, format=self.format)
        self.assertIsInstance(x, sp.spmatrix)
        self.assertEqual(x.format, self.format)
        return x.toarray()


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float64],
    'format': ['csr', 'csc', 'coo'],
}))
@testing.with_requires('scipy')
class TestIdentity(unittest.TestCase):

    @testing.numpy_cupy_allclose(sp_name='sp')
    def test_eye(self, xp, sp):
        x = sp.identity(3, dtype=self.dtype, format=self.format)
        self.assertIsInstance(x, sp.spmatrix)
        self.assertEqual(x.format, self.format)
        return x.toarray()
