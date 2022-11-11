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

    def test_array2string(self):
        a = testing.shaped_arange((2, 3, 4), cupy)
        b = testing.shaped_arange((2, 3, 4), numpy)
        assert cupy.array2string(a) == numpy.array2string(b)

    def test_format_float_positional_python_scalar(self):
        x = 1.0
        assert cupy.format_float_positional(
            x) == numpy.format_float_positional(x)

    def test_format_float_positional(self):
        a = testing.shaped_arange((1,), cupy)
        b = testing.shaped_arange((1,), numpy)
        assert cupy.format_float_positional(
            a) == numpy.format_float_positional(b)

    def test_format_float_scientific(self):
        a = testing.shaped_arange((1,), cupy)
        b = testing.shaped_arange((1,), numpy)
        assert cupy.format_float_scientific(
            a) == numpy.format_float_scientific(b)
