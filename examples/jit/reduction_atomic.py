import cupy
from cupyx import jit


@jit.rawkernel()
def reduction(x, y, size):
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    ntid = jit.blockDim.x * jit.gridDim.x

    value = cupy.float32(0)
    for i in range(tid, size, ntid):
        value += x[i]

    smem = jit.shared_memory(cupy.float32, 1024)
    smem[jit.threadIdx.x] = value

    jit.syncthreads()

    if jit.threadIdx.x == cupy.uint32(0):
        value = cupy.float32(0)
        for i in range(jit.blockDim.x):
            value += smem[i]
        jit.atomic_add(y, 0, value)


size = cupy.uint32(2 ** 22)
x = cupy.random.normal(size=(size,), dtype=cupy.float32)
y = cupy.empty((1,), dtype=cupy.float32)

reduction[64, 1024](x, y, size)

print(y[0])
print(x.sum())
