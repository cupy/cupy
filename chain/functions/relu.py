import numpy
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel

from chain import Function

class ReLU(Function):
    """Rectified Linear Unit."""
    # TODO(beam2d): Implement in-place version.

    def forward_cpu(self, x):
        return numpy.maximum(0, x[0]),

    def forward_gpu(self, x):
        return gpuarray.maximum(0, x[0]),

    def backward_cpu(self, x, gy):
        return gy[0] * (x[0] > 0),

    def backward_gpu(self, x, gy):
        # TODO(beam2d): Unify kernel
        return gy[0] * (x[0] > 0),

def relu(x):
    return ReLU()(x)
