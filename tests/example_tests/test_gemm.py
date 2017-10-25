import unittest

from example_tests import example_test


class TestGEMM(unittest.TestCase):

    def test_sgemm(self):
        example_test.run_example('gemm/sgemm.py')
