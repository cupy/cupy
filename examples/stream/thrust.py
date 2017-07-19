# nvprof --print-gpu-trace python examples/stream/thrust.py
import cupy

x = cupy.array([1, 2, 3])

with cupy.cuda.stream.Stream():
    y = x.sort()

stream = cupy.cuda.stream.Stream()
stream.use()
y = x.sort()
