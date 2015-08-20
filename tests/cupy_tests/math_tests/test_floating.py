import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestFloating(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_binary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a, b)

    def test_signbit(self):
        self.check_unary('signbit')

    def test_copysign(self):
        self.check_binary('copysign')

    @testing.for_float_dtypes(name='ftype')
    @testing.for_dtypes(['i', 'l'], name='itype')
    @testing.numpy_cupy_allclose()
    def test_ldexp(self, xp, ftype, itype):
        a = xp.array([-3, -2, -1, 0, 1, 2, 3], dtype=ftype)
        b = xp.array([-3, -2, -1, 0, 1, 2, 3], dtype=itype)
        return xp.ldexp(a, b)

    @testing.for_float_dtypes()
    def test_frexp(self, dtype):
        numpy_a = numpy.array([-300, -20, -10, -1, 0, 1, 10, 20, 300],
                              dtype=dtype)
        numpy_b, numpy_c = numpy.frexp(numpy_a)

        cupy_a = cupy.array(numpy_a)
        cupy_b, cupy_c = cupy.frexp(cupy_a)

        testing.assert_allclose(cupy_b, numpy_b)
        testing.assert_array_equal(cupy_c, numpy_c)

    def test_nextafter(self):
        self.check_binary('nextafter')
