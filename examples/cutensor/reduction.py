#
# C_{m,v} = alpha * A_{m,h,k,v} + beta * C_{m,v}
#
import numpy
import cupy
from cupy import cutensor
from cupy.cuda import stream

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

alpha = 1.0
beta = 0.1

# rehearsal
c = cutensor.reduction(alpha, a, desc_a, mode_a,
                       beta, c, desc_c, mode_c)

ev_start = stream.Event()
ev_end = stream.Event()
st = stream.Stream()
with st:
    # measurement
    ev_start.record()
    c = cutensor.reduction(alpha, a, desc_a, mode_a,
                           beta, c, desc_c, mode_c)
    ev_end.record()
st.synchronize()

elapsed_ms = stream.get_elapsed_time(ev_start, ev_end)
transfer_byte = a.size * a.itemsize + c.size * c.itemsize
if beta != 0.0:
    transfer_byte += c.size * c.itemsize
gbs = transfer_byte / elapsed_ms / 1e6

print('dtype: {}'.format(numpy.dtype(dtype).name))
print('time (ms): {}'.format(elapsed_ms))
print('effective memory bandwidth (GB/s): {}'.format(gbs))
