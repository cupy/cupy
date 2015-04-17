import ctypes
import libcudnn
import numpy
import pycuda.gpuarray as gpuarray
from chainer import Function, cudnn

_mode = libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_SIGMOID']

class Sigmoid(Function):
    """Logistic sigmoid function."""

    def forward_cpu(self, x):
        self.y = 1 / (1 + numpy.exp(-x[0]))
        return self.y,

    def forward_gpu(self, x):
        handle = cudnn.get_default_handle()
        desc = cudnn.get_tensor_desc(x[0], 1, 1)
        self.y = gpuarray.empty_like(x[0])
        libcudnn.cudnnActivationForward(
            handle, _mode, 1, desc, cudnn.get_ptr(x[0]),
            0, desc, cudnn.get_ptr(self.y))
        return self.y,

    def backward_cpu(self, x, gy):
        return gy[0] * self.y * (1 - self.y),

    def backward_gpu(self, x, gy):
        handle = cudnn.get_default_handle()
        desc = cudnn.get_tensor_desc(self.y, 1, 1)
        gx = gpuarray.empty_like(self.y)
        libcudnn.cudnnActivationBackward(
            handle, _mode, 1, desc, cudnn.get_ptr(self.y),
            desc, cudnn.get_ptr(gy[0]), desc, cudnn.get_ptr(x[0]),
            0, desc, cudnn.get_ptr(gx))
        return gx,

def sigmoid(x):
    return Sigmoid()(x)
