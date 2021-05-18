import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestFormatting(unittest.TestCase):

    def test_array_repr(self):
        a = testing.shaped_arange((2, 3, 4), cupy)
        b = testing.shaped_arange((2, 3, 4), numpy)
        assert cupy.array_repr(a) == numpy.array_repr(b)

    def test_array_str(self):
        a = testing.shaped_arange((2, 3, 4), cupy)
        b = testing.shaped_arange((2, 3, 4), numpy)
        assert cupy.array_str(a) == numpy.array_str(b)
