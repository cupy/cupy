import ctypes
import libcudnn
import numpy
from pycuda import gpuarray
from chainer import Function, cudnn

_mode = libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_TANH']

class Tanh(Function):
    """Hyperbolic tangent function."""

    def forward_cpu(self, x):
        self.y = numpy.tanh(x[0])
        return self.y,

    def forward_gpu(self, x):
        handle = cudnn.get_default_handle()
        desc = cudnn.get_tensor_desc(x[0], 1, 1)
        self.y = gpuarray.empty_like(x[0])
        libcudnn.cudnnActivationForward(
            handle, _mode, 1, desc.value, cudnn.get_ptr(x[0]),
            0, desc.value, cudnn.get_ptr(self.y))
        return self.y,

    def backward_cpu(self, x, gy):
        return gy[0] * (1 - self.y * self.y),

    def backward_gpu(self, x, gy):
        handle = cudnn.get_default_handle()
        desc = cudnn.get_tensor_desc(self.y, 1, 1)
        gx = gpuarray.empty_like(self.y)
        libcudnn.cudnnActivationBackward(
            handle, _mode, 1, desc.value, cudnn.get_ptr(self.y),
            desc.value, cudnn.get_ptr(gy[0]), desc.value, cudnn.get_ptr(x[0]),
            0, desc.value, cudnn.get_ptr(gx))
        return gx,

def tanh(x):
    return Tanh()(x)
