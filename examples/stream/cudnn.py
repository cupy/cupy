# nvprof --print-gpu-trace python examples/stream/cudnn.py
import cupy
import cupy.cuda.cudnn
import cupyx.cudnn

x = cupy.array([1.0, 2.0, 3.0])
mode = cupy.cuda.cudnn.CUDNN_ACTIVATION_RELU
expected = cupyx.cudnn.activation_forward(x, mode)
cupy.cuda.Device().synchronize()

stream = cupy.cuda.stream.Stream()
with stream:
    y = cupyx.cudnn.activation_forward(x, mode)
stream.synchronize()
cupy.testing.assert_array_equal(y, expected)

stream = cupy.cuda.stream.Stream()
stream.use()
y = cupyx.cudnn.activation_forward(x, mode)
stream.synchronize()
cupy.testing.assert_array_equal(y, expected)
