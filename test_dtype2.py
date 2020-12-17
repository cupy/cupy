import numpy as np
import test_dtype


test_dtype.init()
# a = np.empty(10, dtype=test_dtype.complex32)  # this would segfault...
# np.dtype(test_dtype.complex32)  # this segfaults too...
a = np.empty(10, dtype=test_dtype.complex32_dtype)
print(a.dtype)
print(a.size)
print(a.shape)
print(a.ndim)
a[0] = 3
print(a)
