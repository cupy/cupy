# nvprof --print-gpu-trace python examples/stream/cusolver.py
import cupy

x = cupy.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], float)

with cupy.cuda.stream.Stream():
    w, v = cupy.linalg.eigh(x, UPLO='U')

stream = cupy.cuda.stream.Stream()
stream.use()
w, v = cupy.linalg.eigh(x, UPLO='U')
