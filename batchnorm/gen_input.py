import cupy as cp

def gen_input():
    # return cp.random.randn(32, 256, 112, 112)
    # return cp.random.randn(32, 64, 112, 112)
    return cp.random.randn(32, 64, 14, 14)

def gen_test_input():
    return cp.arange(16, dtype=cp.float64).reshape((2, 2, 2, 2))
