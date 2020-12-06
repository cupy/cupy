import cupy as cp
import numpy as np


x = cp.asarray([-1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 11.0])
fx = cp.arange(0, 10, 2, dtype=x.dtype)
fy = cp.sin(fx)
print("x:", x)
print("fx:", fx)
print("fy:", fy)
out = cp.interp(x, fx, fy, left=100, right=200)
print('interp outcome:', out)
x = cp.asnumpy(x)
fx = cp.asnumpy(fx)
fy = cp.asnumpy(fy)
print('expect outcome:', np.interp(x, fx, fy, left=100, right=200))
