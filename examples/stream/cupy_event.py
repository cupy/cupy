# nvprof --print-gpu-trace python examples/stream/cupy_event.py
import cupy

x = cupy.array([1, 2, 3])

start_event = cupy.cuda.stream.Event()
stop_event = cupy.cuda.stream.Event()


def _norm_with_elapsed_time(x):
    start_event.record()
    y = cupy.linalg.norm(x)
    stop_event.record()
    stop_event.synchronize()
    print(cupy.cuda.get_elapsed_time(start_event, stop_event))
    return y


with cupy.cuda.stream.Stream():
    y = _norm_with_elapsed_time(x)

stream = cupy.cuda.stream.Stream()
stream.use()
y = _norm_with_elapsed_time(x)
