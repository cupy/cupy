import unittest

import cupy


class TestGetSize(unittest.TestCase):

    def test_none(self):
        self.assertEqual(cupy._get_size(None), ())

    def check_collection(self, a):
        self.assertEqual(cupy._get_size(a), tuple(a))

    def test_list(self):
        self.check_collection([1, 2, 3])

    def test_tuple(self):
        self.check_collection((1, 2, 3))

    def test_int(self):
        self.assertEqual(cupy._get_size(1), (1,))

    def test_float(self):
        with self.assertRaises(ValueError):
            cupy._get_size(1.0)
