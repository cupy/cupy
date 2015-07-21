import numpy

from chainer import cuda
from chainer import cudnn
from chainer import function
from chainer.utils import type_check

if cudnn.available:
    from chainer.cudnn import libcudnn
    _mode = libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_SIGMOID']


class Sigmoid(function.Function):

    """Logistic sigmoid function."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == numpy.float32)

    def forward_cpu(self, x):
        self.y = 1 / (1 + numpy.exp(-x[0]))
        return self.y,

    def forward_gpu(self, x):
        self.y = cuda.empty_like(x[0])
        if cudnn.enabled and self.use_cudnn:
            handle = cudnn.get_default_handle()
            desc = cudnn.get_tensor_desc(x[0], 1, 1)
            libcudnn.cudnnActivationForward(
                handle, _mode, 1, desc.value, cudnn.get_ptr(x[0]),
                0, desc.value, cudnn.get_ptr(self.y))
        else:
            cuda.elementwise(
                'float* y, const float* x', 'y[i] = 1 / (1 + __expf(-x[i]))',
                'sigmoid_fwd')(self.y, x[0])
        return self.y,

    def backward_cpu(self, x, gy):
        return gy[0] * self.y * (1 - self.y),

    def backward_gpu(self, x, gy):
        gx = cuda.empty_like(x[0])
        if cudnn.enabled and self.use_cudnn:
            handle = cudnn.get_default_handle()
            desc = cudnn.get_tensor_desc(self.y, 1, 1)
            libcudnn.cudnnActivationBackward(
                handle, _mode, 1, desc.value, cudnn.get_ptr(self.y),
                desc.value, cudnn.get_ptr(
                    gy[0]), desc.value, cudnn.get_ptr(x[0]),
                0, desc.value, cudnn.get_ptr(gx))
        else:
            cuda.elementwise(
                'float* gx, const float* y, const float* gy',
                'gx[i] = gy[i] * y[i] * (1 - y[i])',
                'sigmoid_bwd')(gx, self.y, gy[0])
        return gx,


def sigmoid(x, use_cudnn=True):
    """Elementwise sigmoid logistic function :math:`f(x)=(1 + \\exp(-x))^{-1}`.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If True and CuDNN is enabled, then this function uses
            CuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Sigmoid(use_cudnn)(x)
