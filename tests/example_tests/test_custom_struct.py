import re
import unittest

from example_tests import example_test


class TestCustomStruct(unittest.TestCase):

    def test_builtin_vectors(self):
        output = example_test.run_example('custom_struct/builtin_vectors.py')
        assert re.match(
            r"Kernel output matches expected value.",
            output.decode('utf-8'),
        )

    def test_packed_matrix(self):
        output = example_test.run_example('custom_struct/packed_matrix.py')
        assert re.match(
            r"Kernel output matches expected value for type 'float'.\r?\n"
            r"Kernel output matches expected value for type 'double'.",
            output.decode('utf-8'),
        )

    def test_complex_struct(self):
        output = example_test.run_example('custom_struct/complex_struct.py')
        assert re.match(
            r"Overall structure itemsize: \d+ bytes\r?\n"
            r"Structure members itemsize: \[(\s*\d+){5}]\r?\n"
            r"Structure members offsets: \[(\s*\d+){5}]\r?\n"
            r"Complex structure value:\r?\n"
            r"\s+\[.*\]\r?\n"
            r"Kernel output matches expected value.",
            output.decode('utf-8'),
        )
