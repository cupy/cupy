import numpy
import pycuda.gpuarray as gpuarray
from chain import Function

class Softmax(Function):
    """Softmax activation function."""

    def forward_cpu(self, x):
        y = x[0] - numpy.amax(x[0], axis=1, keepdims=True)
        numpy.exp(y, out=y)
        y /= y.sum(axis=1, keepdims=True)
        return y,

    def backward_cpu(self, x, gy):
        y = self.outputs[0].data
        dx = y * gy[0]
        sumdx = dx.sum(axis=1, keepdims=True)
        dx -= y * sumdx
        return dx,

    # TODO(beam2d): GPU implementation

def softmax(x):
    return Softmax()(x)
