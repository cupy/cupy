# nvprof --print-gpu-trace python examples/stream/cublas.py
import cupy

x = cupy.array([1, 2, 3])
y = cupy.array([[1], [2], [3]])

with cupy.cuda.stream.Stream():
    z = cupy.matmul(x, y)

stream = cupy.cuda.stream.Stream()
stream.use()
z = cupy.matmul(x, y)
