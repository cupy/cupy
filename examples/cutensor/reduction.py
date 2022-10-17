#
# C_{m,v} = alpha * A_{m,h,k,v} + beta * C_{m,v}
#
import numpy
import cupy
from cupyx import cutensor
import cupyx.time


dtype = numpy.float32

mode_a = ('m', 'h', 'k', 'v')
mode_c = ('m', 'v')

extent = {'m': 196, 'h': 256, 'k': 64, 'v': 64}

a = cupy.random.random([extent[i] for i in mode_a])
c = cupy.random.random([extent[i] for i in mode_c])
a = a.astype(dtype)
c = c.astype(dtype)

desc_a = cutensor.create_tensor_descriptor(a)
desc_c = cutensor.create_tensor_descriptor(c)

mode_a = cutensor.create_mode(*mode_a)
mode_c = cutensor.create_mode(*mode_c)
alpha = 1.0
beta = 0.1

perf = cupyx.time.repeat(
    cutensor.reduction,
    (alpha, a, desc_a, mode_a, beta, c, desc_c, mode_c),
    n_warmup=1, n_repeat=5)

transfer_byte = a.size * a.itemsize + c.size * c.itemsize
if beta != 0.0:
    transfer_byte += c.size * c.itemsize
elapsed = perf.gpu_times.mean()
gbs = transfer_byte / elapsed / 1e9

print('dtype: {}'.format(numpy.dtype(dtype).name))
print(perf)
print('effective memory bandwidth (GB/s): {}'.format(gbs))
