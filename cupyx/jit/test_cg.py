import cupy as cp
import cupyx.jit as jit


@jit.rawkernel()
def test_grid(x):
    y = jit.cg.this_grid()

a = cp.random.random(100, dtype=cp.float64)
test_grid[1, 32](a)
