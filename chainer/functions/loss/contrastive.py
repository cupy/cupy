import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Contrastive(function.Function):

    """Contrastive loss function."""

    def __init__(self, margin):
        self.margin = float(margin)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)

        x0_type, x1_type, y_type = in_types
        type_check.expect(
            x0_type.dtype == numpy.float32,
            x1_type.dtype == numpy.float32,
            x0_type.shape == x1_type.shape,
            x1_type.shape[0] == y_type.shape[0],
            x0_type.ndim == 2,
            x1_type.ndim == 2,
            y_type.ndim == 1
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x0, x1, y = inputs

        self.diff = x0 - x1  # N x 2
        self.dist_sq = xp.sum(self.diff ** 2, axis=1)  # N
        self.dist = xp.sqrt(self.dist_sq)
        self.mdist = self.margin - self.dist
        dist = xp.maximum(self.mdist, 0)
        loss = y * self.dist_sq + (1 - y) * dist * dist
        loss = xp.sum(loss) / 2.0 / x0.shape[0]

        return xp.array(loss, dtype=xp.float32),

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        x0, x1, y = inputs

        y = xp.vstack((y, y)).T
        alpha = gy[0] / y.shape[0]
        dist = xp.vstack((self.dist, self.dist)).T
        # similar pair
        gx0 = alpha * y * self.diff
        # dissimilar pair
        mdist = xp.vstack((self.mdist, self.mdist)).T
        mdist_p = xp.array(self.mdist > 0, dtype=xp.int32)
        mdist_p = xp.vstack((mdist_p, mdist_p)).T
        gx0 += alpha * (1 - y) * mdist_p * mdist * -(self.diff / dist)
        gx0 = gx0.astype(xp.float32)

        return gx0, -gx0, None


def contrastive(x0, x1, y, margin=1):
    """Computes contrastive loss.
    It takes a variable pair and a label as inputs. The label is 1 when those
    two input variables are similar, or 0 when they are dissimilar. Let
    :math:`N` and :math:`K` denote mini-batchsize and the dimension of input
    variables, respectively, the shape of both input variables should be
    (N, K).
    .. math::
        L = \\frac{1}{2N} \\left( \\sum_{n=1}^N y_n d_n
            + (1 - y_n) \\max ({\\rm margin} - \\sqrt{d_n}, 0)^2 \\right)
    where :math:`N` denotes the mini-batch size, and
    :math:`d_n = \\| {\\bf x_0}_n - {\\bf x_1}_n \\|_2`, and
    :math:`{\\bf x_0}_n` means the n-th K-dimensional vector in a mini-batch.
    Args:
        x0 (~chainer.Variable): The first input variable.
        x1 (~chainer.Variable): The second input variable.
        y (~chainer.Variable): Labels. All values should be 0 or 1.
        margin (int): A parameter for contrastive loss.
    Returns:
        ~chainer.Varible: A variable holding a scalar that is the loss value
            calculated by the above equation.
    """
    return Contrastive(margin)(x0, x1, y)
