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

        mod = jit.transpile(
            kernel, options=('-std=c++11',),
            name_expressions=['kernel<int*,long long>'])
        x = cupy.arange(6, dtype=numpy.int32)
        ker = mod.get_function('kernel<int*,long long>')
        ker((2,), (3,), (x, 2))
        testing.assert_allclose(x, cupy.arange(6, dtype=numpy.int32) + 2)
