import pytest

import cupy
from cupy.cuda import runtime
from cupyx import jit


@pytest.mark.skipif(runtime.is_hip, reason="not supported on HIP")
class TestCooperativeGroups:

    def test_thread_block_group(self):
        @jit.rawkernel()
        def test_thread_block(x):
            g = jit.cg.this_thread_block()
            if g.thread_rank() == 0:
                x[0] += 101
            if g.thread_rank() == 1:
                x[1] = g.size()
            # test dim3
            if g.thread_rank() == 2:
                g_idx = g.group_index()
                x[2], x[3], x[4] = g_idx.x, g_idx.y, g_idx.z
            if g.thread_rank() == 3:
                t_idx = g.thread_index()
                x[5], x[6], x[7] = t_idx.x, t_idx.y, t_idx.z
            if g.thread_rank() == 4:
                g_dim = g.group_dim()
                x[8], x[9], x[10] = g_dim.x, g_dim.y, g_dim.z
            g.sync()

        x = cupy.empty((16,), dtype=cupy.int64)
        x[:] = -1
        test_thread_block[1, 32](x)
        assert x[0] == 100
        assert x[1] == 32
        assert (x[2], x[3], x[4]) == (0, 0, 0)
        assert (x[5], x[6], x[7]) == (3, 0, 0)
        assert (x[8], x[9], x[10]) == (32, 1, 1)
        assert (x[11:] == -1).all()

    @pytest.mark.skipif(
        runtime.runtimeGetVersion() < 11060,
        reason='not supported until CUDA 11.6')
    def test_thread_block_group_num_threads_dim_threads(self):
        @jit.rawkernel()
        def test_thread_block(x):
            g = jit.cg.this_thread_block()
            if g.thread_rank() == 0:
                x[0] = g.num_threads()
            if g.thread_rank() == 1:
                d_th = g.dim_threads()
                x[1], x[2], x[3] = d_th.x, d_th.y, d_th.z
            g.sync()

        x = cupy.empty((16,), dtype=cupy.int64)
        x[:] = -1
        test_thread_block[1, 32](x)
        assert x[0] == 32
        assert (x[1], x[2], x[3]) == (32, 1, 1)
        assert (x[4:] == -1).all()

    @pytest.mark.skipif(runtime.deviceGetAttribute(
        runtime.cudaDevAttrCooperativeLaunch, 0) == 0,
        reason='cooperative launch is not supported on device 0')
    def test_grid_group(self):
        @jit.rawkernel()
        def test_grid(x):
            g = jit.cg.this_grid()
            if g.thread_rank() == 0:
                x[0] = g.is_valid()
            if g.thread_rank() == 1:
                x[1] = g.size()
            if g.thread_rank() == 32:  # on the 2nd group
                # Note: this is not yet possible...
                # x[2], x[3], x[4] == g.group_dim()
                g_dim = g.group_dim()
                x[2], x[3], x[4] = g_dim.x, g_dim.y, g_dim.z
            g.sync()  # this should just work!

        x = cupy.empty((16,), dtype=cupy.uint64)
        x[:] = -1  # = 2**64-1
        test_grid[2, 32](x)
        assert x[0] == 1
        assert x[1] == 64
        assert (x[2], x[3], x[4]) == (2, 1, 1)
        assert (x[5:] == 2**64-1).all()

    @pytest.mark.skipif(
        runtime.runtimeGetVersion() < 11060,
        reason='not supported until CUDA 11.6')
    @pytest.mark.skipif(runtime.deviceGetAttribute(
        runtime.cudaDevAttrCooperativeLaunch, 0) == 0,
        reason='cooperative launch is not supported on device 0')
    def test_grid_group_cu116_new_APIs(self):
        @jit.rawkernel()
        def test_grid(x):
            g = jit.cg.this_grid()
            if g.thread_rank() == 1:
                x[1] = g.num_threads()
            if g.thread_rank() == 32:
                g_dim = g.dim_blocks()
                x[2], x[3], x[4] = g_dim.x, g_dim.y, g_dim.z
            if g.thread_rank() == 33:  # on the 2nd block
                x[5] = g.block_rank()
            if g.thread_rank() == 2:
                x[6] = g.num_blocks()
            if g.thread_rank() == 34:  # on the 2nd block
                b_idx = g.block_index()
                x[7], x[8], x[9] = b_idx.x, b_idx.y, b_idx.z
            g.sync()  # this should just work!

        x = cupy.empty((16,), dtype=cupy.uint64)
        x[:] = -1  # = 2**64-1
        test_grid[2, 32](x)
        assert x[1] == 64
        assert (x[2], x[3], x[4]) == (2, 1, 1)
        assert x[5] == 1
        assert x[6] == 2
        assert (x[7], x[8], x[9]) == (1, 0, 0)
        assert (x[10:] == 2**64-1).all()
