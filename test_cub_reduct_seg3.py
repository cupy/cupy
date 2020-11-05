import cupy as cp
import numpy as np
import ctypes
from cupyx.time import repeat


#cp.core._accelerator.set_routine_accelerators(['cub'])
#cp.core._accelerator.set_reduction_accelerators([])
cp.core._accelerator.set_routine_accelerators([])
cp.core._accelerator.set_reduction_accelerators(['cub'])



#axes = [(0,), (1,), (2,),
#        (0, 1), (0, 2), (1, 2),
#        (0, 1, 2)]
shape = (512, 512, 512)
axes=[(2,)]
#axes = [(1,)]
#shape = (2, 3, 4)

a = cp.random.random(shape, dtype=cp.float32)
a_np = cp.asnumpy(a)
#a = cp.arange(np.prod(shape), dtype=cp.float32).reshape(shape)
#print(a)

for axis in axes:
    #print(f'{axis}:\n', repeat(cp.sum, (a, axis), n_repeat=20))
    #print(f'{axis}:\n', repeat(np.sum, (a_np, axis), n_repeat=20))
    out_cp = cp.sum(a, axis)
    out_np = np.sum(a_np, axis)
    if not np.allclose(cp.asnumpy(out_cp), out_np, rtol=1E-4):
        print("cupy:", out_cp)
        print("numpy:", out_np)
