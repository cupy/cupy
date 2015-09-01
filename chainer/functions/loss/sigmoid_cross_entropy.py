import numpy

from chainer import cuda
from chainer import function
from chainer.functions import sigmoid
from chainer.utils import type_check


class SigmoidCrossEntropy(function.Function):

    """Sigmoid activation followed by a sigmoid cross entropy loss."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.int32,
            x_type.shape == t_type.shape
        )

    def forward_cpu(self, inputs):
        x, t = inputs
        self.y, = sigmoid.Sigmoid().forward_cpu((x,))
        # stable computation of the cross entropy.
        loss = -numpy.sum(
            x * (t - (x >= 0)) - numpy.log1p(numpy.exp(-numpy.abs(x))))
        return numpy.array(loss / t.shape[0], dtype=x.dtype),

    def forward_gpu(self, inputs):
        x, t = inputs
        self.y, = sigmoid.Sigmoid(self.use_cudnn).forward_gpu((x,))
        loss = cuda.reduce(
            'T x, S t, T inv_cnt', 'T out',
            'x * (t - (x >= 0)) - log1p(exp(-fabs(x)))',
            'a + b', 'out = a * inv_cnt', 0,
            'sigmoid_crossent_fwd')(x, t, -1.0 / t.shape[0])
        return loss,

    def backward_cpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        gx = gloss * (self.y - t.astype(self.y.dtype)) / t.shape[0]
        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        t, gloss = inputs[1], grad_outputs[0]
        gx = cuda.elementwise(
            'T y, S t, raw T gloss, T inv_cnt', 'T gx',
            'gx = gloss[0] * inv_cnt * (y - t)',
            'sigmoid_crossent_bwd')(self.y, t, gloss, 1.0 / t.shape[0])
        return gx, None


def sigmoid_cross_entropy(x, t, use_cudnn=True):
    """Computes cross entropy loss for sigmoid activations.

    Args:
        x (Variable): A variable object holding a matrix whose (i, j)-th
            element indicates the unnormalized log probability of the j-th unit
            at the i-th example.
        t (Variable): A variable object holding an int32 vector of groundtruth
            binary labels.

    Returns:
        Variable: A variable object holding a scalar array of the cross entropy
            loss.

    .. note::

       This function is differentiable only by ``x``.

    """
    return SigmoidCrossEntropy(use_cudnn)(x, t)
