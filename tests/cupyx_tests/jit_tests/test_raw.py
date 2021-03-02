import unittest
import numpy

import cupy
from cupyx import jit
from cupy import testing


class TestRaw(unittest.TestCase):

    def test_raw_kick_one_kernel(self):
        @jit.rawkernel()
        def f(x, y):
            y[0] = x[0]

        x = cupy.array([10], dtype=numpy.int32)
        y = cupy.array([20], dtype=numpy.int32)
        f((1,), (1,), (x, y))
        assert int(y[0]) == 10

    def test_raw_elementwise_single_op(self):
        @jit.rawkernel()
        def f(x, y):
            tid = jit.threadIdx.x + jit.blockDim.x * jit.blockIdx.x
            y[tid] = x[tid]

        x = testing.shaped_random((30,), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((30,), dtype=numpy.int32, seed=1)
        f((5,), (6,), (x, y))
        assert bool((x == y).all())

    def test_raw_elementwise_loop(self):
        @jit.rawkernel()
        def f(x, y, size):
            tid = jit.threadIdx.x + jit.blockDim.x * jit.blockIdx.x
            ntid = jit.blockDim.x * jit.gridDim.x
            for i in range(tid, size, ntid):
                y[i] = x[i]

        x = testing.shaped_random((1024,), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((1024,), dtype=numpy.int32, seed=1)
        f((5,), (6,), (x, y, numpy.uint32(1024)))
        assert bool((x == y).all())
