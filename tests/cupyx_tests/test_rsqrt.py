import unittest

import numpy

import cupy
from cupy import testing
import cupyx


@testing.gpu
class TestRsqrt(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    def test_rsqrt(self, dtype):
        # Adding 1.0 to avoid division by zero.
        a = testing.shaped_arange((2, 3), numpy, dtype) + 1.0
        out = cupyx.rsqrt(cupy.array(a))
        # numpy.sqrt is broken in numpy<1.11.2
        testing.assert_allclose(out, 1.0 / numpy.sqrt(a))
