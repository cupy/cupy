#
# C_{m,u,n,v} = alpha * A_{m,h,k,n} * B_{u,k,v,h} + beta * C_{m,u,n,v}
#
import numpy
import cupy
from cupy import cutensor
from cupy.cuda import stream

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

alpha = 1.1
beta = 1.0

# rehearsal
c = cutensor.contraction(alpha, a, desc_a, mode_a, b, desc_b, mode_b,
                         beta, c, desc_c, mode_c)

ev_start = stream.Event()
ev_end = stream.Event()
st = stream.Stream()
with st:
    # measurement
    ev_start.record()
    c = cutensor.contraction(alpha, a, desc_a, mode_a, b, desc_b, mode_b,
                             beta, c, desc_c, mode_c)
    ev_end.record()
st.synchronize()

elapsed_ms = stream.get_elapsed_time(ev_start, ev_end)
total_flops = 2 * numpy.prod(numpy.array(list(extent.values())))

print('dtype: {}'.format(numpy.dtype(dtype).name))
print('time (ms): {}'.format(elapsed_ms))
print('GFLOPS: {}'.format(total_flops / elapsed_ms / 1e6))
