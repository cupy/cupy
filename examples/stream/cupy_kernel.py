# nvprof --print-gpu-trace python examples/stream/cupy_kernel.py
import cupy

x = cupy.array([1, 2, 3])
expected = cupy.linalg.norm(x)
cupy.cuda.Device().synchronize()

stream = cupy.cuda.stream.Stream()
with stream:
    y = cupy.linalg.norm(x)
stream.synchronize()
cupy.testing.assert_array_equal(y, expected)

stream = cupy.cuda.stream.Stream()
stream.use()
y = cupy.linalg.norm(x)
stream.synchronize()
cupy.testing.assert_array_equal(y, expected)
