import unittest

from cupy import cuda


class TestCusolver(unittest.TestCase):

    def test_cusolver_enabled(self):
        self.assertEqual(cuda.runtime.runtimeGetVersion() >= 8000,
                         cuda.cusolver_enabled)
