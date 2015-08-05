import ctypes
import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _algorithm = libcudnn.CUDNN_SOFTMAX_ACCURATE
    _mode = libcudnn.CUDNN_SOFTMAX_MODE_INSTANCE


class Softmax(function.Function):

    """Softmax activation function."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 2,
        )

    def check_type_backward(self, in_types, out_types):
        type_check.expect(
            in_types.size() == 1,
            out_types.size() == 1,
        )
        x_type, = in_types
        y_type, = out_types

        type_check.expect(
            y_type.ndim == 2,

            y_type.shape[0] == x_type.shape[0],
            y_type.shape[1] == x_type.shape[1],
        )

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        if xp != numpy and cuda.cudnn_enabled and self.use_cudnn:
            self.y = cuda.empty_like(x[0])
            handle = cudnn.get_handle()
            x_mat = x[0].reshape(x[0].shape[0], -1, 1, 1)
            desc = cudnn.create_tensor_descriptor(x_mat)
            libcudnn.softmaxForward(
                handle, _algorithm, _mode, ctypes.c_float(1), desc.value,
                x[0].data.ptr, ctypes.c_float(0), desc.value, self.y.data.ptr)
        else:
            self.y = x[0] - x[0].max(axis=1, keepdims=True)
            xp.exp(self.y, out=self.y)
            self.y /= self.y.sum(axis=1, keepdims=True)
        return self.y,

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        if xp != numpy and cuda.cudnn_enabled and self.use_cudnn:
            handle = cudnn.get_handle()
            gx = cuda.empty_like(x[0])
            x_mat = x[0].reshape(x[0].shape[0], -1, 1, 1)
            desc = cudnn.create_tensor_descriptor(x_mat)
            libcudnn.softmaxBackward(
                handle, _algorithm, _mode, ctypes.c_float(1), desc.value,
                self.y.data.ptr, desc.value, gy[0].data.ptr, ctypes.c_float(0),
                desc.value, gx.data.ptr)
        else:
            gx = self.y * gy[0]
            gx -= self.y * gx.sum(axis=1, keepdims=True)

        return gx,


def softmax(x, use_cudnn=True):
    """Channelwise softmax function.

    This function only accepts a two dimensional input array, and computes its
    softmax along the second axis. For each index :math:`i, j` of the input
    matrix :math:`x`, it computes
    :math:`f_{ij}(x)={\\exp(x_{ij}) \\over \\sum_j \\exp(x_{ij})}`.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If True and CuDNN is enabled, then this function uses
            CuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Softmax(use_cudnn)(x)
