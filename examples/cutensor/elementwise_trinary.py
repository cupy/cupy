#
# D_{x,y,z} = alpha * A_{z,y,x} + beta * B_{y,z,x} + gamma * C_{x,y,z}
#
import numpy
import cupy
from cupyx import cutensor
import cupyx.time


dtype = numpy.float32

mode_a = ('z', 'y', 'x')
mode_b = ('y', 'z', 'x')
mode_c = ('x', 'y', 'z')

extent = {'x': 400, 'y': 200, 'z': 300}

a = cupy.random.random([extent[i] for i in mode_a])
b = cupy.random.random([extent[i] for i in mode_b])
c = cupy.random.random([extent[i] for i in mode_c])
a = a.astype(dtype)
b = b.astype(dtype)
c = c.astype(dtype)

alpha = 1.1
beta = 1.2
gamma = 1.3

perf = cupyx.time.repeat(
    cutensor.elementwise_trinary,
    (alpha, a, mode_a,
     beta,  b, mode_b,
     gamma, c, mode_c),
    n_warmup=1, n_repeat=5)

itemsize = numpy.dtype(dtype).itemsize
transfer_byte = a.size * itemsize
if alpha != 0.0:
    transfer_byte += a.size * itemsize
if beta != 0.0:
    transfer_byte += b.size * itemsize
if gamma != 0.0:
    transfer_byte += c.size * itemsize
elapsed = perf.gpu_times.mean()
gbs = transfer_byte / elapsed / 1e9

print('dtype: {}'.format(numpy.dtype(dtype).name))
print(perf)
print('effective memory bandwidth (GB/s): {}'.format(gbs))
