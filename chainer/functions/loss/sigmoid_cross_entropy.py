import numpy

from chainer import cuda
from chainer import function
from chainer.functions import sigmoid
from chainer.utils import type_check


class SigmoidCrossEntropy(function.Function):

    """Sigmoid activation followed by a sigmoid cross entropy loss."""

    ignore_label = -1

    def __init__(self, use_cudnn=True, normalize=True):
        self.use_cudnn = use_cudnn
        self.normalize = normalize

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.int32,
            x_type.shape == t_type.shape
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        self.ignore_mask = (t != self.ignore_label)
        if self.normalize:
            count = int(self.ignore_mask.sum())
        else:
            count = x.shape[0]
        self.count = count if count > 0 else 1
        # stable computation of the cross entropy.
        loss = -xp.sum(
            self.ignore_mask * (x * (t - (x >= 0)) -
                                xp.log1p(xp.exp(-xp.abs(x)))))
        return xp.array(loss / self.count, dtype=x.dtype),

    def backward(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        y, = sigmoid.Sigmoid(self.use_cudnn).forward((x,))
        dtype = y.dtype
        gx = (gloss * self.ignore_mask * (y - t.astype(dtype)) /
              dtype.type(self.count))
        return gx, None


def sigmoid_cross_entropy(x, t, use_cudnn=True, normalize=True):
    """Computes cross entropy loss for sigmoid activations.

    Args:
        x (Variable): A variable object holding a matrix whose (i, j)-th
            element indicates the unnormalized log probability of the j-th unit
            at the i-th example.
        t (Variable): Variable holding an int32 vector of groundtruth labels.
            If ``t[i] == -1``, correspondig ``x[i]`` is ignored.
            Loss becomes to zero if all groundtruth labels are ``-1``.
        normalize (bool): Variable holding a boolean value which
            determines the normalization constant. If true, this function
            normalizes the cross entropy loss across all instances. If else,
            it only normalizes along a batch size.

    Returns:
        Variable: A variable object holding a scalar array of the cross entropy
            loss.

    .. note::

       This function is differentiable only by ``x``.

    """
    return SigmoidCrossEntropy(use_cudnn, normalize)(x, t)
