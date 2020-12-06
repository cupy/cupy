import cupy as cp
import numpy as np


x = cp.asarray([3.5, 4.5, 5.5])
fx = cp.arange(0, 10, 2, dtype=x.dtype)
fy = cp.sin(fx)
out = cp.interp(x, fx, fy)
print('interp outcome:', out)
print('expect outcome:', cp.sin(x))
