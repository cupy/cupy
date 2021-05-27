import cupy
from cupyx import jit


@jit.rawkernel()
def reduction(x, y, size):
    tid = jit.threadIdx.x
    ntid = jit.blockDim.x

    value = cupy.float32(0)
    for i in range(tid, size, ntid):
        value += x[i]

    smem = jit.shared_memory(cupy.float32, 1024)
    smem[tid] = value

    jit.syncthreads()

    if tid == cupy.uint32(0):
        value = cupy.float32(0)
        for i in range(ntid):
            value += smem[i]
        y[0] = value


size = cupy.uint32(2 ** 22)
x = cupy.random.normal(size=(size,), dtype=cupy.float32)
y = cupy.empty((1,), dtype=cupy.float32)

reduction[1, 1024](x, y, size)

print(y[0])
print(x.sum())
