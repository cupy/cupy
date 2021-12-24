import unittest
import numpy

import pytest

from cupyx import jit
from cupy import testing


class TestDeviceFunction(unittest.TestCase):

    def test_device_function(self):
        @jit.rawkernel()
        def f(x, y, z):
            tid = jit.threadIdx.x
            z[tid] = g(x[tid], y[tid]) + x[tid] + y[tid]

        def g(x, y):
            x += 1
            y += 1
            return x + y
        
        x = testing.shaped_random((30,), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((30,), dtype=numpy.int32, seed=1)
        z = testing.shaped_random((30,), dtype=numpy.int32, seed=2)
        f((1,), (30,), (x, y, z))
        assert (z == (x + y + 1) * 2).all()
