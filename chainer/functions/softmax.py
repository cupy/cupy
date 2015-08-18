import ctypes
import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _algorithm = libcudnn.CUDNN_SOFTMAX_ACCURATE
    _mode = libcudnn.CUDNN_SOFTMAX_MODE_CHANNEL


class Softmax(function.Function):

    """Softmax activation function."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim > 1,
        )

    def forward(self, x):
        xp = cuda.get_array_module(*x)
        if xp != numpy and cuda.cudnn_enabled and self.use_cudnn:
            handle = cudnn.get_handle()
            x_cube = x[0].reshape(x[0].shape[:2] + (-1, 1))
            desc = cudnn.create_tensor_descriptor(x_cube)
            self.y = xp.empty_like(x[0])
            libcudnn.softmaxForward(
                handle, _algorithm, _mode, ctypes.c_float(1), desc.value,
                x_cube.data.ptr, ctypes.c_float(0), desc.value,
                self.y.data.ptr)
        else:
            self.y = x[0] - x[0].max(axis=1, keepdims=True)
            xp.exp(self.y, out=self.y)
            self.y /= self.y.sum(axis=1, keepdims=True)

        return self.y,

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        if xp != numpy and cuda.cudnn_enabled and self.use_cudnn:
            handle = cudnn.get_handle()
            gx = xp.empty_like(x[0])
            gx_cube = gx.reshape(gx.shape[:2] + (-1, 1))
            desc = cudnn.create_tensor_descriptor(gx_cube)
            libcudnn.softmaxBackward(
                handle, _algorithm, _mode, ctypes.c_float(1), desc.value,
                self.y.data.ptr, desc.value, gy[0].data.ptr, ctypes.c_float(0),
                desc.value, gx.data.ptr)
        else:
            gx = self.y * gy[0]
            sumdx = gx.sum(axis=1, keepdims=True)
            gx -= self.y * sumdx

        return gx,


def softmax(x, use_cudnn=True):
    """Channelwise softmax function.

    This function computes its softmax along the second axis. Let
    :math:`x = (x_1, x_2, \\dots, x_d)^{\\top}` be the d dimensional index
    array and :math:`f(x)` be the d dimensional input array. For each index
    :math:`x` of the input array :math:`f(x)`, it computes the probability
    :math:`p(x)` defined as
    :math:`p(x) = {\\exp(f(x)) \\over \\sum_{x_2} \\exp(f(x))}`.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If True and CuDNN is enabled, then this function uses
            CuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Softmax(use_cudnn)(x)
