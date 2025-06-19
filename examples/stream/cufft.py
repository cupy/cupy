# nvprof --print-gpu-trace python examples/stream/cufft.py
import cupy

x = cupy.array([1, 0, 3, 0, 5, 0, 7, 0, 9], dtype=float)
expected_f = cupy.fft.fft(x)
cupy.cuda.Device().synchronize()

stream = cupy.cuda.stream.Stream()
with stream:
    f = cupy.fft.fft(x)
stream.synchronize()
cupy.testing.assert_array_equal(f, expected_f)

stream = cupy.cuda.stream.Stream()
stream.use()
f = cupy.fft.fft(x)
stream.synchronize()
cupy.testing.assert_array_equal(f, expected_f)
