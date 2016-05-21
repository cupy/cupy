import numpy

from chainer import cuda
from chainer import function
from chainer import utils
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
        zero = utils.force_type(x[0].dtype, 0)
        return utils.force_array(numpy.maximum(zero, x[0])),

    def forward_gpu(self, x):
        if cuda.cudnn_enabled and self.use_cudnn:
            y = cudnn.activation_forward(x[0], _mode)
            self.y = y
        else:
            y = cuda.cupy.maximum(x[0].dtype.type(0), x[0])
        return y,

    def backward_cpu(self, x, gy):
        return utils.force_array(gy[0] * (x[0] > 0)),

    def backward_gpu(self, x, gy):
        if cuda.cudnn_enabled and self.use_cudnn:
            gx = cudnn.activation_backward(x[0], self.y, gy[0], _mode)
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
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return ReLU(use_cudnn)(x)
