import numpy as np
import test_dtype


test_dtype.init()
a = np.empty(10, dtype=test_dtype.complex32)
x = np.dtype(test_dtype.complex32)
y = a[0]
print(type(y))
print(y)
print(a.dtype)
print(a.size)
print(a.shape)
print(a.ndim)
# a[0] = 3
#a * 3
print(a)
