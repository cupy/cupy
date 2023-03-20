#
# C_{m,u,n,v} = alpha * A_{m,h,k,n} * B_{u,k,v,h} + beta * C_{m,u,n,v}
#
import numpy
import cupy
from cupyx import cutensor
import cupyx.time


dtype = numpy.float32

mode_a = ('m', 'h', 'k', 'n')
mode_b = ('u', 'k', 'v', 'h')
mode_c = ('m', 'u', 'n', 'v')

extent = {'m': 96, 'n': 96, 'u': 96, 'v': 64, 'h': 64, 'k': 64}

a = cupy.random.random([extent[i] for i in mode_a])
b = cupy.random.random([extent[i] for i in mode_b])
c = cupy.random.random([extent[i] for i in mode_c])
a = a.astype(dtype)
b = b.astype(dtype)
c = c.astype(dtype)

desc_a = cutensor.create_tensor_descriptor(a)
desc_b = cutensor.create_tensor_descriptor(b)
desc_c = cutensor.create_tensor_descriptor(c)

mode_a = cutensor.create_mode(*mode_a)
mode_b = cutensor.create_mode(*mode_b)
mode_c = cutensor.create_mode(*mode_c)
alpha = 1.1
beta = 1.0

perf = cupyx.time.repeat(
    cutensor.contraction,
    (alpha, a, desc_a, mode_a, b, desc_b, mode_b, beta, c, desc_c, mode_c),
    n_warmup=1, n_repeat=5)

total_flops = 2 * numpy.prod(numpy.array(list(extent.values())))
elapsed = perf.gpu_times.mean()

print('dtype: {}'.format(numpy.dtype(dtype).name))
print(perf)
print('GFLOPS: {}'.format(total_flops / elapsed / 1e9))
