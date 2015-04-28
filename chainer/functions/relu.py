import ctypes
import libcudnn
import numpy
from chainer import cuda, cudnn, Function

_mode = libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_RELU']

class ReLU(Function):
    """Rectified Linear Unit."""
    # TODO(beam2d): Implement in-place version.

    def forward_cpu(self, x):
        return numpy.maximum(0, x[0]),

    def forward_gpu(self, x):
        handle = cudnn.get_default_handle()
        desc = cudnn.get_tensor_desc(x[0], 1, 1)
        self.y = cuda.empty_like(x[0])
        libcudnn.cudnnActivationForward(
            handle, _mode, 1, desc.value, cudnn.get_ptr(x[0]),
            0, desc.value, cudnn.get_ptr(self.y))
        return self.y,

    def backward_cpu(self, x, gy):
        return gy[0] * (x[0] > 0),

    def backward_gpu(self, x, gy):
        handle = cudnn.get_default_handle()
        desc = cudnn.get_tensor_desc(self.y, 1, 1)
        gx = cuda.empty_like(self.y)
        libcudnn.cudnnActivationBackward(
            handle, _mode, 1, desc.value, cudnn.get_ptr(self.y),
            desc.value, cudnn.get_ptr(gy[0]), desc.value, cudnn.get_ptr(x[0]),
            0, desc.value, cudnn.get_ptr(gx))
        return gx,

def relu(x):
    return ReLU()(x)
