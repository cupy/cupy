"""Sample input exercising the migration analyzer's rule coverage."""
import numpy as np
import cupy as cp
from numpy import zeros as mkzeros
from numpy.linalg import inv

# unsupported dtype
a = mkzeros((100, 100), dtype=np.uint64)

# discouraged dtype
b = np.ones(1000, dtype=np.float64)
c = np.asarray([1, 2, 3], dtype="f8")

# dynamic dtype -> UNKNOWN
dtype = get_dtype_from_config()
d = np.zeros(10, dtype=dtype)

# matmul dtype restriction
w = np.matmul(x, y)                 # x is int32 -> unsupported
v = np.tensordot(p, q, axes=1)

# power with bool
r = np.power(flag, 3)

# argument / semantic rules from api_diff
s = np.take(g, idx, axis=1)
t = np.take(g, idx, mode="wrap")
k = np.sort(g, kind="quicksort")
part = np.partition(g, 3)
ss = np.searchsorted(g, v)
pad = np.pad(g, 2, mode="symmetric")

# CPU-fallback IO
data = np.load("checkpoint.npy")

# removed numpy 1.x aliases -> auto-safe rename
total = np.product(g)
running = np.cumproduct(g)

# from-imported module call
precision = 1.0 / inv(m)

# cupy-specific rules
fused = cp.fusion()
am = cp.argmax(m, axis=0)

# deprecated scalar alias / unsigned
u = np.zeros(5, dtype=np.uint32)
obj = np.array([1, "a"], dtype=object)
