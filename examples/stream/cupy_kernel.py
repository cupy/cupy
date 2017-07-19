# nvprof --print-gpu-trace python examples/stream/cupy_kernel.py
import cupy

x = cupy.array([1, 2, 3])

with cupy.cuda.stream.Stream():
    y = cupy.linalg.norm(x)

stream = cupy.cuda.stream.Stream()
stream.use()
y = cupy.linalg.norm(x)
