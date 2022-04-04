import cupy as cp
import cupyx.jit as jit


@jit.rawkernel()
def test_thread_block(x):
    y = jit.cg.this_thread_block()

a = cp.random.random(100, dtype=cp.float64)
test_thread_block[1, 32](a)
