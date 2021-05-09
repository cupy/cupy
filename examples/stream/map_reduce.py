import cupy
import time

device = cupy.cuda.Device()
memory_pool = cupy.cuda.MemoryPool()
cupy.cuda.set_allocator(memory_pool.malloc)
rand = cupy.random.RandomState(seed=1)

n = 10
zs = []
map_streams = []
stop_events = []
reduce_stream = cupy.cuda.stream.Stream()
for i in range(n):
    map_streams.append(cupy.cuda.stream.Stream())

start_time = time.time()

# Map
for stream in map_streams:
    with stream:
        x = rand.normal(size=(1, 1024**2))
        y = rand.normal(size=(1024**2, 1))
        z = cupy.matmul(x, y)
        zs.append(z)
    stop_event = stream.record()
    stop_events.append(stop_event)

# Block the `reduce_stream` until all events occur. This does not block host.
# This is not required when reduction is performed in the default (Stream.null)
# stream unless streams are created with `non_blocking=True` flag.
for i in range(n):
    reduce_stream.wait_event(stop_events[i])

# Reduce
with reduce_stream:
    z = sum(zs)

device.synchronize()
elapsed_time = time.time() - start_time
print('elapsed time', elapsed_time)
print('total bytes', memory_pool.total_bytes())

# Free all blocks in the memory pool of streams
for stream in map_streams:
    memory_pool.free_all_blocks(stream=stream)
print('total bytes', memory_pool.total_bytes())
