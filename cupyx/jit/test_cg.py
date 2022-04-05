import cupy as cp
import cupyx.jit as jit


@jit.rawkernel()
def test_thread_block(x):
    y = jit.cg.this_thread_block()
    y.sync()
    a = y.group_index()
    i = a.x

@jit.rawkernel()
def test_grid(x):
    y = jit.cg.this_grid()
    z = y.is_valid()
    y.sync()

a = cp.random.random(100, dtype=cp.float64)
test_thread_block[1, 32](a)
#test_grid[1, 32](a)
