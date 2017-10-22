# nvprof --print-gpu-trace python examples/stream/thrust.py
import cupy

x = cupy.array([1, 3, 2])
expected = x.sort()
cupy.cuda.Device().synchronize()

stream = cupy.cuda.stream.Stream()
with stream:
    y = x.sort()
stream.synchronize()
cupy.testing.assert_array_equal(y, expected)

stream = cupy.cuda.stream.Stream()
stream.use()
y = x.sort()
stream.synchronize()
cupy.testing.assert_array_equal(y, expected)
