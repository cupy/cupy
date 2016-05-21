import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _mode = libcudnn.CUDNN_ACTIVATION_SIGMOID


class Sigmoid(function.Function):

    """Logistic sigmoid function."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == numpy.float32)

    def forward_cpu(self, x):
        self.y = numpy.tanh(x[0] * 0.5) * 0.5 + 0.5
        return self.y,

    def forward_gpu(self, inputs):
        x = inputs[0]
        if cuda.cudnn_enabled and self.use_cudnn:
            self.y = cuda.cupy.cudnn.activation_forward(x, _mode)
        else:
            self.y = cuda.elementwise(
                'T x', 'T y', 'y = 1 / (1 + exp(-x))',
                'sigmoid_fwd')(x)
        return self.y,

    def backward_cpu(self, x, gy):
        return gy[0] * self.y * (1 - self.y),

    def backward_gpu(self, inputs, grads):
        x = inputs[0]
        gy = grads[0]
        if cuda.cudnn_enabled and self.use_cudnn:
            gx = cuda.cupy.cudnn.activation_backward(x, self.y, gy, _mode)
        else:
            gx = cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = gy * y * (1 - y)',
                'sigmoid_bwd')(self.y, gy)
        return gx,


def sigmoid(x, use_cudnn=True):
    """Elementwise sigmoid logistic function :math:`f(x)=(1 + \\exp(-x))^{-1}`.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Sigmoid(use_cudnn)(x)
