import unittest

import numpy

import cupy
from cupy import testing
from cupyx import jit


@testing.gpu
class TestJit(unittest.TestCase):

    def test_assign(self):

        @jit.cuda_function()
        def kernel(x, y):
            tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
            x[tid] += y
        x = cupy.arange(6, dtype=numpy.int32)
        kernel(x, y)
        testing.assert_allclose(x, cupy.arange(6, dtype=numpy.int32) + 2)
