#
# D_{x,y,z} = alpha * A_{z,y,x} + beta * B_{y,z,x} + gamma * C_{x,y,z}
#
import numpy
import cupy
from cupy import cutensor
from cupy.cuda import stream

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

desc_a = cutensor.create_tensor_descriptor(a)
desc_b = cutensor.create_tensor_descriptor(b)
desc_c = cutensor.create_tensor_descriptor(c)

alpha = 1.1
beta = 1.2
gamma = 1.3

# rehearsal
d = cutensor.elementwise_trinary(alpha, a, desc_a, mode_a,
                                 beta,  b, desc_b, mode_b,
                                 gamma, c, desc_c, mode_c)

ev_start = stream.Event()
ev_end = stream.Event()
st = stream.Stream()
with st:
    # measurement
    ev_start.record()
    d = cutensor.elementwise_trinary(alpha, a, desc_a, mode_a,
                                     beta,  b, desc_b, mode_b,
                                     gamma, c, desc_c, mode_c)
    ev_end.record()
st.synchronize()

elapsed_ms = stream.get_elapsed_time(ev_start, ev_end)
transfer_byte = d.size * d.itemsize
if alpha != 0.0:
    transfer_byte += a.size * a.itemsize
if beta != 0.0:
    transfer_byte += b.size * b.itemsize
if gamma != 0.0:
    transfer_byte += c.size * c.itemsize
gbs = transfer_byte / elapsed_ms / 1e6

print('dtype: {}'.format(numpy.dtype(dtype).name))
print('time (ms): {}'.format(elapsed_ms))
print('effective memory bandwidth (GB/s): {}'.format(gbs))
