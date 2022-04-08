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

#    def test_grid_group(self):
#        @jit.rawkernel()
#        def test_grid(x):
#            y = jit.cg.this_grid()
#            z = y.is_valid()
#            y.sync()
