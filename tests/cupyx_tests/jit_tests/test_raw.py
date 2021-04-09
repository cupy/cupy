import unittest
import numpy

import pytest

import cupy
from cupyx import jit
from cupy import testing


class TestRaw(unittest.TestCase):

    def test_raw_onw_thread(self):
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

    def test_raw_multidimensional_array(self):
        @jit.rawkernel()
        def f(x, y, n_row, n_col):
            tid = jit.threadIdx.x + jit.blockDim.x * jit.blockIdx.x
            ntid = jit.blockDim.x * jit.gridDim.x
            size = n_row * n_col
            for i in range(tid, size, ntid):
                i_row = i // n_col
                i_col = i % n_col
                y[i_row, i_col] = x[i_row, i_col]

        n, m = numpy.uint32(12), numpy.uint32(13)
        x = testing.shaped_random((n, m), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((n, m), dtype=numpy.int32, seed=1)
        f((5,), (6,), (x, y, n, m))
        assert bool((x == y).all())

    def test_raw_0dim_array(self):
        @jit.rawkernel()
        def f(x, y):
            y[()] = x[()]

        x = testing.shaped_random((), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((), dtype=numpy.int32, seed=1)
        f((1,), (1,), (x, y))
        assert bool((x == y).all())

    def test_syncthreads(self):
        @jit.rawkernel()
        def f(x, y, buf):
            tid = jit.threadIdx.x + jit.threadIdx.y * jit.blockDim.x
            ntid = jit.blockDim.x * jit.blockDim.y
            buf[tid] = x[ntid - tid - 1]
            jit.syncthreads()
            y[tid] = buf[ntid - tid - 1]

        x = testing.shaped_random((1024,), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((1024,), dtype=numpy.int32, seed=1)
        buf = testing.shaped_random((1024,), dtype=numpy.int32, seed=2)
        f((1,), (32, 32), (x, y, buf))
        assert bool((x == y).all())

    def test_shared_memory_static(self):
        @jit.rawkernel()
        def f(x, y):
            tid = jit.threadIdx.x
            ntid = jit.blockDim.x
            bid = jit.blockIdx.x
            i = tid + bid * ntid

            smem = jit.shared_memory(numpy.int32, 32)
            smem[tid] = x[i]
            jit.syncthreads()
            y[i] = smem[ntid - tid - 1]

        x = testing.shaped_random((1024,), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((1024,), dtype=numpy.int32, seed=1)
        f((32,), (32,), (x, y))
        expected = x.reshape(32, 32)[:, ::-1].ravel()
        assert bool((y == expected).all())

    def test_shared_memory_dynamic(self):
        @jit.rawkernel()
        def f(x, y):
            tid = jit.threadIdx.x
            ntid = jit.blockDim.x
            bid = jit.blockIdx.x
            i = tid + bid * ntid

            smem = jit.shared_memory(numpy.int32, None)
            smem[tid] = x[i]
            jit.syncthreads()
            y[i] = smem[ntid - tid - 1]

        x = testing.shaped_random((1024,), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((1024,), dtype=numpy.int32, seed=1)
        f((32,), (32,), (x, y), shared_mem=128)
        expected = x.reshape(32, 32)[:, ::-1].ravel()
        assert bool((y == expected).all())

    def test_raw_grid_block_interface(self):
        @jit.rawkernel()
        def f(x, y, size):
            tid = jit.threadIdx.x + jit.blockDim.x * jit.blockIdx.x
            ntid = jit.blockDim.x * jit.gridDim.x
            for i in range(tid, size, ntid):
                y[i] = x[i]

        x = testing.shaped_random((1024,), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((1024,), dtype=numpy.int32, seed=1)
        f[5, 6](x, y, numpy.uint32(1024))
        assert bool((x == y).all())

    def test_error_msg(self):
        @jit.rawkernel()
        def f(x):
            return unknown_var  # NOQA

        import re
        mes = re.escape('''Unbound name: unknown_var

  @jit.rawkernel()
  def f(x):
>     return unknown_var  # NOQA
''')
        x = cupy.zeros((10,), dtype=numpy.float32)
        with pytest.raises(NameError, match=mes):
            f((1,), (1,), (x,))
