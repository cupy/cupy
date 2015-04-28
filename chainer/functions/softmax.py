import ctypes
import libcudnn
import numpy
from chainer import cuda, cudnn, Function

_algorithm = libcudnn.cudnnSoftmaxAlgorithm['CUDNN_SOFTMAX_ACCURATE']
_mode      = libcudnn.cudnnSoftmaxMode['CUDNN_SOFTMAX_MODE_INSTANCE']

class Softmax(Function):
    """Softmax activation function."""

    def forward_cpu(self, x):
        assert x[0].ndim == 2
        self.y = x[0] - numpy.amax(x[0], axis=1, keepdims=True)
        numpy.exp(self.y, out=self.y)
        self.y /= self.y.sum(axis=1, keepdims=True)
        return self.y,

    def forward_gpu(self, x):
        handle = cudnn.get_default_handle()
        desc = cudnn.get_tensor_desc(x[0], 1, 1)
        self.y = cuda.empty_like(x[0])
        libcudnn.cudnnSoftmaxForward(
            handle, _algorithm, _mode, 1, desc.value, cudnn.get_ptr(x[0]),
            0, desc.value, cudnn.get_ptr(self.y))
        return self.y,

    def backward_cpu(self, x, gy):
        gx = self.y * gy[0]
        sumdx = gx.sum(axis=1, keepdims=True)
        gx -= self.y * sumdx
        return gx,

    def backward_gpu(self, x, gy):
        handle = cudnn.get_default_handle()
        desc = cudnn.get_tensor_desc(x[0], 1, 1)
        gx = cuda.empty_like(x[0])
        libcudnn.cudnnSoftmaxBackward(
            handle, _algorithm, _mode, 1, desc.value, cudnn.get_ptr(self.y),
            desc.value, cudnn.get_ptr(gy[0]), 0, desc.value, cudnn.get_ptr(gx))
        return gx,

def softmax(x):
    return Softmax()(x)
