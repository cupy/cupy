import unittest
import numpy

import pytest

import cupy
import cupyx
from cupyx import jit
from cupy import testing
from cupy.cuda import device
from cupy.cuda import runtime


class TestRaw(unittest.TestCase):

    def test_raw_grid_1D(self):
        @jit.rawkernel()
        def f(arr1, arr2):
            x = jit.grid(1)
            if x < arr1.size:
                arr2[x] = arr1[x]

        x = cupy.arange(10)
        y = cupy.empty_like(x)
        f((1,), (10,), (x, y))
        assert (x == y).all()

    def test_raw_grid_2D(self):
        @jit.rawkernel()
        def f(arr1, arr2, n, m):
            x, y = jit.grid(2)
            # TODO(leofang): make it possible to write this:
            # if x < arr1.shape[0] and y < arr1.shape[1]:
            if x < n and y < m:
                arr2[x, y] = arr1[x, y]

        x = cupy.arange(20).reshape(4, 5)
        y = cupy.empty_like(x)
        f((1,), (4, 5), (x, y, x.shape[0], x.shape[1]))
        assert (x == y).all()

    def test_raw_grid_3D(self):
        @jit.rawkernel()
        def f(arr1, arr2, k, m, n):
            x, y, z = jit.grid(3)
            if x < k and y < m and z < n:
                arr2[x, y, z] = arr1[x, y, z]

        l, m, n = (2, 3, 4)
        x = cupy.arange(24).reshape(l, m, n)
        y = cupy.empty_like(x)
        f(((l+1)//2, (m+1)//2, (n+1)//2), (2, 2, 2), (x, y, l, m, n))
        assert (x == y).all()

    def test_raw_grid_invalid1(self):
        @jit.rawkernel()
        def f():
            x, = jit.grid(1)  # cannot unpack an int

        with pytest.raises(ValueError):
            f((1,), (1,), ())

    def test_raw_grid_invalid2(self):
        @jit.rawkernel()
        def f():
            x = jit.grid(2)
            y = cupy.int64(x)  # <- x is a tuple  # NOQA

        # we don't care the exception type as long as something is raised
        with pytest.raises(Exception):
            f((1,), (1,), ())

    def test_raw_grid_invalid3(self):
        for n in (0, 4, 'abc', [0], (1,)):
            @jit.rawkernel()
            def f():
                x = jit.grid(n)  # n can only be 1, 2, 3 (as int)  # NOQA

            err = ValueError if isinstance(n, int) else TypeError
            with pytest.raises(err):
                f((1,), (1,), ())

    def test_raw_gridsize_1D(self):
        @jit.rawkernel()
        def f(arr):
            x = jit.grid(1)
            gx = jit.gridsize(1)
            if x == 0:
                arr[x] = gx

        x = cupy.zeros(1)
        grid = 8
        block = 10
        f((grid,), (block,), (x,))
        assert x == grid * block

    def test_raw_gridsize_2D(self):
        @jit.rawkernel()
        def f(arr):
            x, y = jit.grid(2)
            gx, gy = jit.gridsize(2)
            if x == 0 and y == 0:
                arr[0] = gx
                arr[1] = gy

        x = cupy.zeros(2)
        grid = (2, 7)
        block = (3, 5)
        f(grid, block, (x,))
        assert x[0] == grid[0] * block[0]
        assert x[1] == grid[1] * block[1]

    def test_raw_gridsize_3D(self):
        @jit.rawkernel()
        def f(arr):
            x, y, z = jit.grid(3)
            gx, gy, gz = jit.gridsize(3)
            if x == 0 and y == 0 and z == 0:
                arr[0] = gx
                arr[1] = gy
                arr[2] = gz

        x = cupy.zeros(3)
        grid = (2, 7, 4)
        block = (3, 5, 11)
        f(grid, block, (x,))
        assert x[0] == grid[0] * block[0]
        assert x[1] == grid[1] * block[1]
        assert x[2] == grid[2] * block[2]

    def test_raw_one_thread(self):
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

    def test_raw_multidimensional_array_with_attr(self):
        @jit.rawkernel()
        def f(x, y):
            tid = jit.threadIdx.x + jit.blockDim.x * jit.blockIdx.x
            ntid = jit.blockDim.x * jit.gridDim.x
            n_col = x.size // len(x)
            for i in range(tid, x.size, ntid):
                i_row = i // n_col
                i_col = i % n_col
                y[i_row, i_col] = x[i_row, i_col]

        n, m = numpy.uint32(12), numpy.uint32(13)
        x = testing.shaped_random((n, m), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((n, m), dtype=numpy.int32, seed=1)
        f((5,), (6,), (x, y))
        assert bool((x == y).all())

    def test_raw_ndim(self):
        @jit.rawkernel()
        def f(x, y):
            y[0] = x.ndim

        x = cupy.empty((1, 1, 1, 1, 1, 1, 1), dtype=numpy.int32)
        y = cupy.zeros((1,), dtype=numpy.int64)
        f((1,), (1,), (x, y))
        assert y.item() == 7

    def test_raw_0dim_array(self):
        @jit.rawkernel()
        def f(x, y):
            y[()] = x[()]

        x = testing.shaped_random((), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((), dtype=numpy.int32, seed=1)
        f((1,), (1,), (x, y))
        assert bool((x == y).all())

    def test_min(self):
        @jit.rawkernel()
        def f(x, y, z, r):
            tid = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
            r[tid] = min(x[tid], y[tid], z[tid])

        x = testing.shaped_random((1024,), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((1024,), dtype=numpy.int32, seed=1)
        z = testing.shaped_random((1024,), dtype=numpy.int32, seed=2)
        r = testing.shaped_random((1024,), dtype=numpy.int32, seed=3)
        f((8,), (128,), (x, y, z, r))
        expected = cupy.minimum(x, cupy.minimum(y, z))
        assert bool((r == expected).all())

    def test_max(self):
        @jit.rawkernel()
        def f(x, y, z, r):
            tid = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
            r[tid] = max(x[tid], y[tid], z[tid])

        x = testing.shaped_random((1024,), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((1024,), dtype=numpy.int32, seed=1)
        z = testing.shaped_random((1024,), dtype=numpy.int32, seed=2)
        r = testing.shaped_random((1024,), dtype=numpy.int32, seed=3)
        f((8,), (128,), (x, y, z, r))
        expected = cupy.maximum(x, cupy.maximum(y, z))
        assert bool((r == expected).all())

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

    def test_syncwarp(self):
        @jit.rawkernel()
        def f(x):
            laneId = jit.threadIdx.x & 0x1f
            if laneId < 16:
                x[laneId] = 1
            else:
                x[laneId] = 2
            jit.syncwarp()

        x = cupy.zeros((32,), dtype=numpy.int32)
        y = cupy.ones_like(x)
        f((1,), (32,), (x,))
        y[16:] += 1
        assert bool((x == y).all())

    def test_syncwarp_mask(self):
        @jit.rawkernel()
        def f(x, m):
            laneId = jit.threadIdx.x & 0x1f
            if laneId < m:
                x[laneId] = 1
                jit.syncwarp(mask=m)

        for mask in (2, 4, 8, 16, 32):
            x = cupy.zeros((32,), dtype=numpy.int32)
            y = cupy.zeros_like(x)
            f((1,), (32,), (x, mask))
            y[:mask] += 1
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

    @staticmethod
    def _check(a, e):
        if a.dtype == numpy.float16:
            testing.assert_allclose(a, e, rtol=3e-2, atol=3e-2)
        else:
            testing.assert_allclose(a, e, rtol=1e-5, atol=1e-5)

    @testing.for_dtypes('iILQfd' if runtime.is_hip else 'iILQefd')
    def test_atomic_add(self, dtype):
        @jit.rawkernel()
        def f(x, index, out):
            tid = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
            jit.atomic_add(out, index[tid], x[tid])

        x = testing.shaped_random((1024,), dtype=dtype, seed=0)
        index = testing.shaped_random(
            (1024,), dtype=numpy.bool_, seed=1).astype(numpy.int32)
        out = cupy.zeros((2,), dtype=dtype)
        f((32,), (32,), (x, index, out))

        expected = cupy.zeros((2,), dtype=dtype)
        cupyx.scatter_add(expected, index, x)
        self._check(out, expected)

    @testing.for_dtypes('iI')
    def test_atomic_sub(self, dtype):
        @jit.rawkernel()
        def f(x, out):
            tid = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
            if tid < x.size:
                jit.atomic_sub(out, 0, x[tid])

        x = cupy.ones((1024,), dtype=dtype)
        out = cupy.sum(x, dtype=dtype).reshape(1,)
        f((32,), (32,), (x, out))
        expected = cupy.zeros_like(out)
        self._check(out, expected)

    @testing.for_dtypes('iILQf')
    def test_atomic_exch(self, dtype):
        @jit.rawkernel()
        def f(x, out):
            tid = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
            if tid < x.size:
                jit.atomic_exch(out, x.size - tid - 1, x[tid])

        x = cupy.arange(1024, dtype=dtype)
        out = cupy.zeros_like(x)
        f((32,), (32,), (x, out))
        expected = x[::-1]
        self._check(out, expected)

    @testing.for_dtypes('iILQ')
    def test_atomic_min(self, dtype):
        @jit.rawkernel()
        def f(x, out):
            tid = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
            if tid < x.size:
                jit.atomic_min(out, tid, x[tid])

        x = cupy.arange(1024, dtype=dtype)
        out = x + 1
        f((32,), (32,), (x, out))  # effectively copy x to out
        self._check(out, x)

    @testing.for_dtypes('iILQ')
    def test_atomic_max(self, dtype):
        @jit.rawkernel()
        def f(x, out):
            tid = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
            if tid < x.size:
                jit.atomic_max(out, tid, x[tid])

        x = cupy.arange(1, 1025, dtype=dtype)
        out = x - 1
        f((32,), (32,), (x, out))  # effectively copy x to out
        self._check(out, x)

    def test_atomic_inc(self):
        dtype = cupy.uint32  # atomic_inc only supports 1 dtype

        @jit.rawkernel()
        def f(x, out):
            tid = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
            if tid < x.size:
                # = 0 if out[tid] >= x[tid] else out[tid] + 1
                jit.atomic_inc(out, tid, x[tid])

        x = cupy.arange(1, 1025, dtype=dtype)
        out = x - 1
        f((32,), (32,), (x, out))
        expected = x
        self._check(out, expected)

    def test_atomic_dec(self):
        dtype = cupy.uint32  # atomic_dec only supports 1 dtype

        @jit.rawkernel()
        def f(x, out):
            tid = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
            if tid < x.size:
                # = x[tid] if (out[tid] == 0 or out[tid] >= x[tid]) \
                # else out[tid] - 1
                jit.atomic_dec(out, tid, x[tid])

        x = cupy.zeros(1024, dtype=dtype)
        out = cupy.arange(1024, dtype=dtype)
        f((32,), (32,), (x, out))
        expected = x
        self._check(out, expected)

    @testing.for_dtypes('iILQ' if runtime.is_hip else 'iHILQ')
    def test_atomic_cas(self, dtype):
        if dtype == cupy.uint16:
            if (
                runtime.is_hip or
                runtime.runtimeGetVersion() < 10010 or
                int(device.get_compute_capability()) < 70
            ):
                self.skipTest('not supported')

        @jit.rawkernel()
        def f(x, y, out):
            tid = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
            if tid < x.size:
                # = y[tid] if out[tid] == x[tid] else out[tid]
                jit.atomic_cas(out, tid, x[tid], y[tid])

        x = cupy.arange(1024, dtype=dtype)
        y = x.copy()
        y[512:] = 0
        out = x.copy()
        out[:512] = 0
        f((32,), (32,), (x, y, out))
        expected = cupy.zeros_like(out)
        self._check(out, expected)

    @testing.for_dtypes('iILQ')
    def test_atomic_and(self, dtype):
        @jit.rawkernel()
        def f(x, out):
            tid = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
            if tid < x.size:
                jit.atomic_and(out, tid, x[tid])

        x = cupy.arange(1024, dtype=dtype)
        out = cupy.ones_like(x)
        f((32,), (32,), (x, out))
        expected = cupy.zeros_like(out)
        expected[1::2] = 1
        self._check(out, expected)

    @testing.for_dtypes('iILQ')
    def test_atomic_or(self, dtype):
        @jit.rawkernel()
        def f(x, out):
            tid = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
            if tid < x.size:
                jit.atomic_or(out, tid, x[tid])

        x = testing.shaped_random((1024,), dtype=dtype, seed=0)
        out = testing.shaped_random((1024,), dtype=dtype, seed=1)
        y = out.copy()
        f((32,), (32,), (x, out))
        expected = x | y
        self._check(out, expected)

    @testing.for_dtypes('iILQ')
    def test_atomic_xor(self, dtype):
        @jit.rawkernel()
        def f(x, out):
            tid = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
            if tid < x.size:
                jit.atomic_xor(out, tid, x[tid])

        @jit.rawkernel()
        def f_no_atomic(x, out):
            tid = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
            if tid < x.size:
                out[tid] = out[tid] ^ x[tid]

        x = testing.shaped_random((1024,), dtype=dtype, seed=0)
        out = testing.shaped_random((1024,), dtype=dtype, seed=1)
        expected = out.copy()
        f((32,), (32,), (x, out))
        f_no_atomic((32,), (32,), (x, expected))
        self._check(out, expected)

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

    # TODO(leofang): test float16 ('e') once cupy/cupy#5346 is resolved
    @testing.for_dtypes('iIlqfd' if runtime.is_hip else 'iIlLqQfd')
    def test_shfl(self, dtype):
        # strictly speaking this function is invalid in Python (see the
        # discussion in cupy/cupy#5340), but it serves for our purpose
        @jit.rawkernel()
        def f(a, b):
            laneId = jit.threadIdx.x & 0x1f
            if laneId == 0:
                value = a
            value = jit.shfl_sync(0xffffffff, value, 0)
            b[laneId] = value

        a = dtype(100)
        b = cupy.empty((32,), dtype=dtype)
        f[1, 32](a, b)
        assert (b == a * cupy.ones((32,), dtype=dtype)).all()

    def test_shfl_width(self):
        @jit.rawkernel()
        def f(a, b, w):
            laneId = jit.threadIdx.x & 0x1f
            value = jit.shfl_sync(0xffffffff, b[jit.threadIdx.x], 0, width=w)
            b[laneId] = value

        c = cupy.arange(32, dtype=cupy.int32)
        for w in (2, 4, 8, 16, 32):
            a = cupy.int32(100)
            b = cupy.arange(32, dtype=cupy.int32)
            f[1, 32](a, b, w)
            c[c % w != 0] = c[c % w == 0]
            assert (b == c).all()

    # TODO(leofang): test float16 ('e') once cupy/cupy#5346 is resolved
    @testing.for_dtypes('iIlqfd' if runtime.is_hip else 'iIlLqQfd')
    def test_shfl_up(self, dtype):
        N = 5

        @jit.rawkernel()
        def f(a):
            value = jit.shfl_up_sync(0xffffffff, a[jit.threadIdx.x], N)
            a[jit.threadIdx.x] = value

        a = cupy.arange(32, dtype=dtype)
        f[1, 32](a)
        expected = [i for i in range(N)] + [i for i in range(32-N)]
        assert(a == cupy.asarray(expected, dtype=dtype)).all()

    # TODO(leofang): test float16 ('e') once cupy/cupy#5346 is resolved
    @testing.for_dtypes('iIlqfd' if runtime.is_hip else 'iIlLqQfd')
    def test_shfl_down(self, dtype):
        N = 5
        # __shfl_down() on HIP does not seem to have the same behavior...
        block = 64 if runtime.is_hip else 32

        @jit.rawkernel()
        def f(a):
            value = jit.shfl_down_sync(0xffffffff, a[jit.threadIdx.x], N)
            a[jit.threadIdx.x] = value

        a = cupy.arange(block, dtype=dtype)
        f[1, block](a)
        expected = [i for i in range(N, block)]
        expected += [(block-N+i) for i in range(N)]
        assert(a == cupy.asarray(expected, dtype=dtype)).all()

    # TODO(leofang): test float16 ('e') once cupy/cupy#5346 is resolved
    @testing.for_dtypes('iIlqfd' if runtime.is_hip else 'iIlLqQfd')
    def test_shfl_xor(self, dtype):
        @jit.rawkernel()
        def f_shfl_xor(a):
            laneId = jit.threadIdx.x & 0x1f
            value = 31 - laneId
            i = 16
            while i >= 1:
                value += jit.shfl_xor_sync(0xffffffff, value, i)
                i //= 2
            a[jit.threadIdx.x] = value

        a = cupy.arange(32, dtype=dtype)
        b = a.copy()
        f_shfl_xor[1, 32](a)
        assert (a == b.sum() * cupy.ones(32, dtype=dtype)).all()

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

    def test_laneid(self):
        @jit.rawkernel()
        def f(arr):
            x = jit.grid(1)
            if x < arr.size:
                arr[x] = jit.laneid()

        N = 64 if runtime.is_hip else 32
        x = cupy.zeros((N*2,), dtype=cupy.uint32)
        f((1,), (N*2,), (x,))
        y = cupy.arange(N*2, dtype=cupy.uint32) % N
        assert (x == y).all()

    def test_warpsize(self):
        @jit.rawkernel()
        def f(arr):
            x = jit.grid(1)
            if x == 0:
                arr[0] = jit.warpsize

        N = 64 if runtime.is_hip else 32
        x = cupy.zeros((1,), dtype=cupy.uint32)
        f((1,), (1,), (x,))
        assert x == N
