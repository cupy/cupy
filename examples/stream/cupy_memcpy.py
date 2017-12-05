# nvprof --print-gpu-trace python examples/stream/cupy_memcpy.py
import cupy
import numpy

pinned_memory_pool = cupy.cuda.PinnedMemoryPool()
cupy.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)


def _pin_memory(array):
    mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
    ret = numpy.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret


x_cpu = numpy.array([1, 2, 3], dtype=numpy.float32)
x_pinned_cpu = _pin_memory(x_cpu)
x_gpu = cupy.core.ndarray((3,), dtype=numpy.float32)
with cupy.cuda.stream.Stream():
    x_gpu.set(x_pinned_cpu)

stream = cupy.cuda.stream.Stream()
stream.use()
x_pinned_cpu = x_gpu.get()
