# nvprof --print-gpu-trace python examples/stream/cusolver.py
import cupy

x = cupy.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], float)
expected_w, expected_v = cupy.linalg.eigh(x, UPLO='U')
cupy.cuda.Device().synchronize()

stream = cupy.cuda.stream.Stream()
with stream:
    w, v = cupy.linalg.eigh(x, UPLO='U')
stream.synchronize()
cupy.testing.assert_array_equal(w, expected_w)
cupy.testing.assert_array_equal(v, expected_v)

stream = cupy.cuda.stream.Stream()
stream.use()
w, v = cupy.linalg.eigh(x, UPLO='U')
stream.synchronize()
cupy.testing.assert_array_equal(w, expected_w)
cupy.testing.assert_array_equal(v, expected_v)
