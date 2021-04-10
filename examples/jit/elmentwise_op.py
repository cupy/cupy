import cupy
from cupyx import jit


@jit.rawkernel()
def elementwise_copy(x, y, size):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.gridDim.x * jit.blockDim.x
    for i in range(tid, size, ntid):
        y[i] = x[i]


size = cupy.uint32(2 ** 22)
x = cupy.random.normal(size=(size,), dtype=cupy.float32)
y = cupy.empty((size,), dtype=cupy.float32)

elementwise_copy((128,), (1024,), (x, y, size))

elementwise_copy[128, 1024](x, y, size)

assert (x == y).all()
