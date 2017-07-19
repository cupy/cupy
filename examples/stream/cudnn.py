# nvprof --print-gpu-trace python examples/stream/cudnn.py
import cupy
import cupy.cudnn

x = cupy.array([1.0, 2.0, 3.0])
mode = cupy.cuda.cudnn.CUDNN_ACTIVATION_RELU

with cupy.cuda.stream.Stream():
    y = cupy.cudnn.activation_forward(x, mode)

stream = cupy.cuda.stream.Stream()
stream.use()
y = cupy.cudnn.activation_forward(x, mode)
