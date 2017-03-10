import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    _mode = libcudnn.CUDNN_ACTIVATION_SIGMOID


class Sigmoid(function.Function):

    """Logistic sigmoid function."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        half = x[0].dtype.type(0.5)
        self.y = utils.force_array(numpy.tanh(x[0] * half) * half + half)
        return self.y,

    def forward_gpu(self, inputs):
        x = inputs[0]
        if (cuda.cudnn_enabled and self.use_cudnn and x.flags.c_contiguous and
                (_cudnn_version >= 3000 or x.dtype != numpy.float16)):
            self.y = cuda.cupy.cudnn.activation_forward(x, _mode)
        else:
            self.y = cuda.elementwise(
                'T x', 'T y', 'y = tanh(x * 0.5) * 0.5 + 0.5',
                'sigmoid_fwd')(x)
        return self.y,

    def backward_cpu(self, x, gy):
        one = x[0].dtype.type(1)
        return utils.force_array(gy[0] * self.y * (one - self.y)),

    def backward_gpu(self, inputs, grads):
        x = inputs[0]
        gy = grads[0]
        if (cuda.cudnn_enabled and self.use_cudnn and x.flags.c_contiguous and
                gy.flags.c_contiguous and
                (_cudnn_version >= 3000 or x.dtype != numpy.float16)):
            gx = cuda.cupy.cudnn.activation_backward(x, self.y, gy, _mode)
        else:
            gx = cuda.elementwise(
                'T y, T gy', 'T gx',
                'gx = gy * y * (1 - y)',
                'sigmoid_bwd')(self.y, gy)
        return gx,


def sigmoid(x, use_cudnn=True):
    """Element-wise sigmoid logistic function.

     .. math:: f(x)=(1 + \\exp(-x))^{-1}.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        It maps the input values into the range of :math:`[0, 1]`.
        >>> x = np.arange(-2, 3, 2).astype('f')
        >>> x
        array([-2.,  0.,  2.], dtype=float32)
        >>> F.sigmoid(x).data
        array([ 0.11920291,  0.5       ,  0.88079709], dtype=float32)

    """
    return Sigmoid(use_cudnn)(x)
