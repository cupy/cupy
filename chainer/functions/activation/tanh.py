import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _mode = libcudnn.CUDNN_ACTIVATION_TANH


class Tanh(function.Function):

    """Hyperbolic tangent function."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == numpy.float32)

    def forward_cpu(self, x):
        self.y = numpy.tanh(x[0])
        return self.y,

    def forward_gpu(self, x):
        self.y = cuda.cupy.empty_like(x[0])
        if cuda.cudnn_enabled and self.use_cudnn:
            dtype = x[0].dtype
            one = numpy.array(1, dtype=dtype).ctypes
            zero = numpy.array(0, dtype=dtype).ctypes
            handle = cudnn.get_handle()
            x_mat = x[0].reshape(x[0].shape[0], -1, 1, 1)
            desc = cudnn.create_tensor_descriptor(x_mat)
            libcudnn.activationForward_v3(
                handle, _mode, one.data, desc.value, x_mat.data.ptr,
                zero.data, desc.value, self.y.data.ptr)
        else:
            cuda.cupy.tanh(x[0], out=self.y)
        return self.y,

    def backward_cpu(self, x, gy):
        return gy[0] * (1 - self.y * self.y),

    def backward_gpu(self, x, gy):
        if cuda.cudnn_enabled and self.use_cudnn:
            gx = cuda.cupy.empty_like(self.y)
            dtype = x[0].dtype
            one = numpy.array(1, dtype=dtype).ctypes
            zero = numpy.array(0, dtype=dtype).ctypes
            handle = cudnn.get_handle()
            y_mat = self.y.reshape(self.y.shape[0], -1, 1, 1)
            desc = cudnn.create_tensor_descriptor(y_mat)
            libcudnn.activationBackward_v3(
                handle, _mode, one.data, desc.value, y_mat.data.ptr,
                desc.value, gy[0].data.ptr, desc.value, x[0].data.ptr,
                zero.data, desc.value, gx.data.ptr)
        else:
            gx = cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = gy * (1 - y * y)',
                'tanh_bwd')(self.y, gy[0])
        return gx,


def tanh(x, use_cudnn=True):
    """Elementwise hyperbolic tangent function.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Tanh(use_cudnn)(x)
