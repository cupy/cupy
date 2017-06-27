import unittest

import numpy

from cupy import testing


@testing.gpu
class TestContent(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_dtypes('efFdD')
    @testing.numpy_cupy_array_equal()
    def check_unary_inf(self, name, xp, dtype):
        a = xp.array([-3, dtype(numpy.inf), -1, -dtype(numpy.inf), 0, 1, 2],
                     dtype=dtype)
        return getattr(xp, name)(a)

    @testing.for_dtypes('efFdD')
    @testing.numpy_cupy_array_equal()
    def check_unary_nan(self, name, xp, dtype):
        if numpy.dtype(dtype).kind == 'c':
            a = xp.array(
                [-3. + 1j, numpy.NAN, -1, numpy.NAN, 0, numpy.NAN,
                 dtype(numpy.inf)],
                dtype=dtype)
        else:
            a = xp.array(
                [-3, numpy.NAN, -1, numpy.NAN, 0, numpy.NAN, dtype('inf')],
                dtype=dtype)
        return getattr(xp, name)(a)

    def test_isfinite(self):
        self.check_unary_inf('isfinite')

    def test_isinf(self):
        self.check_unary_inf('isinf')

    def test_isnan(self):
        self.check_unary_nan('isnan')
