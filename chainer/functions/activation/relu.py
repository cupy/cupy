import ctypes
import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _mode = libcudnn.CUDNN_ACTIVATION_RELU


def _as4darray(arr):
    if arr.ndim == 0:
        return arr.reshape(1, 1, 1, 1)
    else:
        return arr.reshape(arr.shape[0], -1, 1, 1)


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
        zero = utils.force_type(x[0].dtype, 0)
        return utils.force_array(numpy.maximum(zero, x[0])),

    def forward_gpu(self, x):
        y = cuda.empty_like(x[0])
        if cuda.cudnn_enabled and self.use_cudnn:
            handle = cudnn.get_handle()
            desc = cudnn.create_tensor_descriptor(_as4darray(x[0]))
            libcudnn.activationForward(
                handle, _mode, ctypes.c_float(1), desc.value, x[0].data.ptr,
                ctypes.c_float(0), desc.value, y.data.ptr)
            self.y = y
        else:
            y = cuda.cupy.maximum(x[0].dtype.type(0), x[0])
        return y,

    def backward_cpu(self, x, gy):
        return utils.force_array(gy[0] * (x[0] > 0)),

    def backward_gpu(self, x, gy):
        if cuda.cudnn_enabled and self.use_cudnn:
            gx = cuda.empty_like(x[0])
            handle = cudnn.get_handle()
            desc = cudnn.create_tensor_descriptor(_as4darray(self.y))
            libcudnn.activationBackward(
                handle, _mode, ctypes.c_float(1), desc.value, self.y.data.ptr,
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
