import ctypes
import libcudnn
import numpy
import pycuda.gpuarray as gpuarray
from chain import Function, cudnn

_algorithm = libcudnn.cudnnSoftmaxAlgorithm['CUDNN_SOFTMAX_ACCURATE']
_mode      = libcudnn.cudnnSoftmaxMode['CUDNN_SOFTMAX_MODE_INSTANCE']

class Softmax(Function):
    """Softmax activation function."""

    def forward_cpu(self, x):
        assert x[0].ndim == 2
        y = x[0] - numpy.amax(x[0], axis=1, keepdims=True)
        numpy.exp(y, out=y)
        y /= y.sum(axis=1, keepdims=True)
        return y,

    def forward_gpu(self, x):
        handle = cudnn.get_default_handle()
        desc = cudnn.get_tensor_desc(x[0], 1, 1)
        y = gpuarray.empty_like(x[0])
        libcudnn.cudnnSoftmaxForward(
            handle, _algorithm, _mode,
            1, desc, cudnn.get_ptr(x[0]), 0, desc, cudnn.get_ptr(y))
        return y,

    def backward_cpu(self, x, gy):
        y = self.outputs[0].data
        gx = y * gy[0]
        sumdx = gx.sum(axis=1, keepdims=True)
        gx -= y * sumdx
        return gx,

    def backward_gpu(self, x, gy):
        y, gy = self.outputs[0].data, gy[0]
        handle = cudnn.get_default_handle()
        desc = cudnn.get_tensor_desc(x[0], 1, 1)
        gx = gpuarray.empty_like(x[0])
        libcudnn.cudnnSoftmaxBackward(
            handle, _algorithm, _mode, 1, desc, cudnn.get_ptr(y),
            desc, cudnn.get_ptr(gy), 0, desc, cudnn.get_ptr(gx))
        return gx,

def softmax(x):
    return Softmax()(x)
