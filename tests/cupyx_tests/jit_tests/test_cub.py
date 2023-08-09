import cupy
from cupy import testing
from cupy_backends.cuda.api import runtime
from cupyx import jit


class TestCubWarpReduce:

    @testing.for_all_dtypes(no_bool=True)
    def test_sum(self, dtype):

        @jit.rawkernel()
        def warp_reduce_sum(x, y):
            WarpReduce = jit.cub.WarpReduce[dtype]
            temp_storage = jit.shared_memory(
                dtype=WarpReduce.TempStorage, size=1)
            i, j = jit.blockIdx.x, jit.threadIdx.x
            value = x[i, j]
            aggregater = WarpReduce(temp_storage[0])
            aggregate = aggregater.Sum(value)
            if j == 0:
                y[i] = aggregate

        warp_size = 64 if runtime.is_hip else 32
        h, w = (32, warp_size)
        x = testing.shaped_random((h, w), dtype=dtype)
        y = testing.shaped_random((h,), dtype=dtype)
        expected = cupy.asnumpy(x).sum(axis=-1, dtype=dtype)

        warp_reduce_sum[h, w](x, y)
        testing.assert_allclose(y, expected, rtol=1e-6)

    @testing.for_all_dtypes(no_bool=True)
    def test_reduce_sum(self, dtype):

        @jit.rawkernel()
        def warp_reduce_sum(x, y):
            WarpReduce = jit.cub.WarpReduce[dtype]
            temp_storage = jit.shared_memory(
                dtype=WarpReduce.TempStorage, size=1)
            i, j = jit.blockIdx.x, jit.threadIdx.x
            value = x[i, j]
            aggregater = WarpReduce(temp_storage[0])
            aggregate = aggregater.Reduce(value, jit.cub.Sum())
            if j == 0:
                y[i] = aggregate

        warp_size = 64 if runtime.is_hip else 32
        h, w = (32, warp_size)
        x = testing.shaped_random((h, w), dtype=dtype)
        y = testing.shaped_random((h,), dtype=dtype)
        expected = cupy.asnumpy(x).sum(axis=-1, dtype=dtype)

        warp_reduce_sum[h, w](x, y)
        testing.assert_allclose(y, expected, rtol=1e-6)

    @testing.for_all_dtypes(no_bool=True)
    def test_reduce_max(self, dtype):

        @jit.rawkernel()
        def warp_reduce_max(x, y):
            WarpReduce = jit.cub.WarpReduce[dtype]
            temp_storage = jit.shared_memory(
                dtype=WarpReduce.TempStorage, size=1)
            i, j = jit.blockIdx.x, jit.threadIdx.x
            value = x[i, j]
            aggregater = WarpReduce(temp_storage[0])
            aggregate = aggregater.Reduce(value, jit.cub.Max())
            if j == 0:
                y[i] = aggregate

        warp_size = 64 if runtime.is_hip else 32
        h, w = (32, warp_size)
        x = testing.shaped_random((h, w), dtype=dtype)
        y = testing.shaped_random((h,), dtype=dtype)
        expected = cupy.asnumpy(x).max(axis=-1)

        warp_reduce_max[h, w](x, y)
        testing.assert_allclose(y, expected, rtol=1e-6)


class TestCubBlockReduce:

    @testing.for_all_dtypes(no_bool=True)
    def test_sum(self, dtype):

        @jit.rawkernel()
        def block_reduce_sum(x, y):
            BlockReduce = jit.cub.BlockReduce[dtype, 256]
            temp_storage = jit.shared_memory(
                dtype=BlockReduce.TempStorage, size=1)
            i, j = jit.blockIdx.x, jit.threadIdx.x
            value = x[i, j]
            aggregater = BlockReduce(temp_storage[0])
            aggregate = aggregater.Sum(value)
            if j == 0:
                y[i] = aggregate

        h, w = (32, 256)
        x = testing.shaped_random((h, w), dtype=dtype)
        y = testing.shaped_random((h,), dtype=dtype)
        expected = cupy.asnumpy(x).sum(axis=-1, dtype=dtype)

        block_reduce_sum[h, w](x, y)
        testing.assert_allclose(y, expected, rtol=1e-6)

    @testing.for_all_dtypes(no_bool=True)
    def test_reduce_min(self, dtype):

        @jit.rawkernel()
        def block_reduce_min(x, y):
            BlockReduce = jit.cub.BlockReduce[dtype, 256]
            temp_storage = jit.shared_memory(
                dtype=BlockReduce.TempStorage, size=1)
            i, j = jit.blockIdx.x, jit.threadIdx.x
            value = x[i, j]
            aggregater = BlockReduce(temp_storage[0])
            aggregate = aggregater.Reduce(value, jit.cub.Min())
            if j == 0:
                y[i] = aggregate

        h, w = (32, 256)
        x = testing.shaped_random((h, w), dtype=dtype)
        y = testing.shaped_random((h,), dtype=dtype)
        expected = cupy.asnumpy(x).min(axis=-1)

        block_reduce_min[h, w](x, y)
        testing.assert_allclose(y, expected, rtol=1e-6)
