import unittest

import cupy
from cupy import testing

@testing.gpu
class TestAll(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        self.x = cupy.arange(10) % 2 == 1

    def test_simple(self):
        self.assertFalse(cupy.all(self.x))

@testing.gpu
class TestAny(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        self.x = cupy.arange(10) % 2 == 1

    def test_simple(self):
        self.assertTrue(cupy.any(self.x))

