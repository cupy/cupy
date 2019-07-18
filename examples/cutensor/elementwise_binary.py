#
# D_{x,y,z} = alpha * A_{z,y,x} + gamma * C_{x,y,z}
#
import numpy
import cupy
from cupy import cutensor
from cupy.cuda import stream

dtype = numpy.float32

mode_a = ('z', 'y', 'x')
mode_c = ('x', 'y', 'z')

extent = {'x': 400, 'y': 200, 'z': 300}

a = cupy.random.random([extent[i] for i in mode_a])
c = cupy.random.random([extent[i] for i in mode_c])
a = a.astype(dtype)
c = c.astype(dtype)

desc_a = cutensor.create_tensor_descriptor(a)
desc_c = cutensor.create_tensor_descriptor(c)

alpha = 1.1
gamma = 1.3

# rehearsal
d = cutensor.elementwise_binary(alpha, a, desc_a, mode_a,
                                gamma, c, desc_c, mode_c)

ev_start = stream.Event()
ev_end = stream.Event()
st = stream.Stream()
with st:
    # measurement
    ev_start.record()
    d = cutensor.elementwise_binary(alpha, a, desc_a, mode_a,
                                    gamma, c, desc_c, mode_c)
    ev_end.record()
st.synchronize()

elapsed_ms = stream.get_elapsed_time(ev_start, ev_end)
transfer_byte = d.size * d.itemsize
if alpha != 0.0:
    transfer_byte += a.size * a.itemsize
if gamma != 0.0:
    transfer_byte += c.size * c.itemsize
gbs = transfer_byte / elapsed_ms / 1e6

print('dtype: {}'.format(numpy.dtype(dtype).name))
print('time (ms): {}'.format(elapsed_ms))
print('effective memory bandwidth (GB/s): {}'.format(gbs))
