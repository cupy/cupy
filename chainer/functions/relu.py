import ctypes
import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _mode = libcudnn.CUDNN_ACTIVATION_RELU


class ReLU(function.Function):

    """Rectified Linear Unit."""
    # TODO(beam2d): Implement in-place version.

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype == numpy.float32
        )

    def forward_cpu(self, x):
        return numpy.maximum(0, x[0]),

    def forward_gpu(self, x):
        y = cuda.empty_like(x[0])
        if cuda.cudnn_enabled and self.use_cudnn:
            handle = cudnn.get_handle()
            x_mat = x[0].reshape(x[0].shape[0], -1, 1, 1)
            desc = cudnn.create_tensor_descriptor(x_mat)
            libcudnn.activationForward(
                handle, _mode, ctypes.c_float(1), desc.value, x_mat.data.ptr,
                ctypes.c_float(0), desc.value, y.data.ptr)
            self.y = y
        else:
            y = cuda.cupy.maximum(0, x[0])
        return y,

    def backward_cpu(self, x, gy):
        return gy[0] * (x[0] > 0),

    def backward_gpu(self, x, gy):
        if cuda.cudnn_enabled and self.use_cudnn:
            gx = cuda.empty_like(x[0])
            handle = cudnn.get_handle()
            y_mat = self.y.reshape(self.y.shape[0], -1, 1, 1)
            desc = cudnn.create_tensor_descriptor(y_mat)
            libcudnn.activationBackward(
                handle, _mode, ctypes.c_float(1), desc.value, y_mat.data.ptr,
                desc.value, gy[0].data.ptr, desc.value, x[0].data.ptr,
                ctypes.c_float(0), desc.value, gx.data.ptr)
        else:
            gx = cuda.elementwise(
                'T x, T gy', 'T gx',
                'gx = x > 0 ? gy : 0',
                'relu_bwd')(x[0], gy[0])
        return gx,


def relu(x, use_cudnn=True):
    """Rectified Linear Unit function :math:`f(x)=\\max(0, x)`.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If True and CuDNN is enabled, then this function uses
            CuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return ReLU(use_cudnn)(x)
