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


SIZE = 1024 * 1024
x_cpu_src = numpy.arange(SIZE, dtype=numpy.float32)
x_gpu_src = cupy.arange(SIZE, dtype=numpy.float32)


# synchronous
stream = cupy.cuda.Stream.null
start = stream.record()
x_gpu_dst = cupy.empty(x_cpu_src.shape, x_cpu_src.dtype)
x_gpu_dst.set(x_cpu_src)
x_cpu_dst = x_gpu_src.get()
end = stream.record()

print('Synchronous Device to Host / Host to Device (ms)')
print(cupy.cuda.get_elapsed_time(start, end))


# asynchronous
x_gpu_dst = cupy.empty(x_cpu_src.shape, x_cpu_src.dtype)
x_cpu_dst = numpy.empty(x_gpu_src.shape, x_gpu_src.dtype)

x_pinned_cpu_src = _pin_memory(x_cpu_src)
x_pinned_cpu_dst = _pin_memory(x_cpu_dst)

with cupy.cuda.stream.Stream() as stream_htod:
    start = stream_htod.record()
    x_gpu_dst.set(x_pinned_cpu_src)
    with cupy.cuda.stream.Stream() as stream_dtoh:
        x_gpu_src.get(out=x_pinned_cpu_dst)
        stream_dtoh.synchronize()
    stream_htod.synchronize()
    end = stream_htod.record()

print('Asynchronous Device to Host / Host to Device (ms)')
print(cupy.cuda.get_elapsed_time(start, end))
