import numpy

from chainer import cuda
from chainer import function
from chainer.functions import sigmoid
from chainer import utils
from chainer.utils import type_check


class SigmoidCrossEntropy(function.Function):

    """Sigmoid activation followed by a sigmoid cross entropy loss."""

    ignore_label = -1

    def __init__(self, use_cudnn=True, normalize=True, keepdims=False):
        self.use_cudnn = use_cudnn
        self.normalize = normalize
        self.keepdims = keepdims

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

        # stable computation of the cross entropy.
        loss = -(
            self.ignore_mask *
            (x * (t - (x >= 0)) - xp.log1p(xp.exp(-xp.abs(x)))))

        if self.keepdims:
            return utils.force_array(loss.astype(x.dtype)),

        if self.normalize:
            count = xp.maximum(1, self.ignore_mask.sum())
        else:
            count = max(1, len(x))
        self.count = count

        return utils.force_array(
            xp.divide(xp.sum(loss), self.count, dtype=x.dtype)),

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        gloss = grad_outputs[0]
        y, = sigmoid.Sigmoid(self.use_cudnn).forward((x,))
        if self.keepdims:
            gx = (gloss * self.ignore_mask * (y - t)).astype(y.dtype)
        else:
            gx = xp.divide(
                gloss * self.ignore_mask * (y - t), self.count,
                dtype=y.dtype)
        return gx, None


def sigmoid_cross_entropy(
        x, t, use_cudnn=True, normalize=True, keepdims=False):
    """Computes cross entropy loss for pre-sigmoid activations.

    Args:
        x (Variable): A variable object holding a matrix whose (i, j)-th
            element indicates the unnormalized log probability of the j-th unit
            at the i-th example.
        t (Variable): Variable holding an int32 vector of ground truth labels.
            If ``t[i] == -1``, corresponding ``x[i]`` is ignored.
            Loss is zero if all ground truth labels are ``-1``.
        normalize (bool): Variable holding a boolean value which
            determines the normalization constant. If true, this function
            normalizes the cross entropy loss across all instances. If else,
            it only normalizes along a batch size.
        keepdims (bool): Variable holding a boolean value which
            determines whether to keep the shape of the input.
            If true, this function computes cross entropy for each instance
            and does not normalize it (``normalize`' option is ignored).
            If else, it computes the sum of cross entropy
            and normalize it according to ``normalize`` option.

    Returns:
        Variable: A variable object holding an array of the cross entropy.
            If ``keepdim`` is true, the shape is same as ``x``.
            If else, it is a scalar array.

    .. note::

       This function is differentiable only by ``x``.

    """
    return SigmoidCrossEntropy(use_cudnn, normalize, keepdims)(x, t)
