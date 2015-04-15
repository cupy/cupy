import ctypes
import libcudnn
import numpy
import pycuda.gpuarray as gpuarray
from chain import Function, cudnn

_mode = libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_TANH']

class Tanh(Function):
    """Hyperbolic tangent function."""

    def forward_cpu(self, x):
        return numpy.tanh(x[0]),

    def forward_gpu(self, x):
        handle = cudnn.get_default_handle()
        desc = cudnn.get_tensor_desc(x[0], 1, 1)
        y = gpuarray.empty_like(x[0])
        libcudnn.cudnnActivationForward(
            handle, _mode, 1, desc, cudnn.get_ptr(x[0]),
            0, desc, cudnn.get_ptr(y))
        return y,

    def backward_cpu(self, x, gy):
        y = self.outputs[0].data
        return gy[0] * (1 - y * y),

    def backward_gpu(self, x, gy):
        y, gy = self.outputs[0].data, gy[0]
        handle = cudnn.get_default_handle()
        desc = cudnn.get_tensor_desc(y, 1, 1)
        gx = gpuarray.empty_like(y)
        libcudnn.cudnnActivationBackward(
            handle, _mode, 1, desc, cudnn.get_ptr(y), desc, cudnn.get_ptr(gy),
            desc, cudnn.get_ptr(x[0]), 0, desc, cudnn.get_ptr(gx))
        return gx,

def tanh(x):
    return Tanh()(x)
