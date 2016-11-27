import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Contrastive(function.Function):

    """Contrastive loss function."""

    def __init__(self, margin):
        if margin <= 0:
            raise ValueError("margin should be positive value.")
        self.margin = margin

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)

        x0_type, x1_type, y_type = in_types
        type_check.expect(
            x0_type.dtype == numpy.float32,
            x1_type.dtype == numpy.float32,
            y_type.dtype == numpy.int32,
            x0_type.shape == x1_type.shape,
            x1_type.shape[0] == y_type.shape[0],
            x1_type.shape[0] > 0,
            x0_type.ndim == 2,
            x1_type.ndim == 2,
            y_type.ndim == 1
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x0, x1, y = inputs

        self.diff = x0 - x1
        self.dist_sq = xp.sum(self.diff ** 2, axis=1)
        self.dist = xp.sqrt(self.dist_sq)
        self.mdist = self.margin - self.dist
        dist = xp.maximum(self.mdist, 0)
        loss = y * self.dist_sq + (1 - y) * dist * dist
        loss = xp.sum(loss) / 2.0 / x0.shape[0]

        return xp.array(loss, dtype=xp.float32),

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        x0, x1, y = inputs

        x_dim = x0.shape[1]
        y = xp.repeat(y[:, None], x_dim, axis=1)
        alpha = gy[0] / y.shape[0]
        dist = xp.repeat(self.dist[:, None], x_dim, axis=1)
        # avoid division by zero
        dist = xp.maximum(dist, 1e-8)
        # similar pair
        gx0 = alpha * y * self.diff
        # dissimilar pair
        mdist = xp.repeat(self.mdist[:, None], x_dim, axis=1)
        mdist_p = xp.array(mdist > 0, dtype=xp.int32)
        gx0 += alpha * (1 - y) * mdist_p * mdist * -(self.diff / dist)
        gx0 = gx0.astype(xp.float32)

        return gx0, -gx0, None


def contrastive(x0, x1, y, margin=1):
    """Computes contrastive loss.

    It takes a pair of variables and a label as inputs. The label is 1 when
    those two input variables are similar, or 0 when they are dissimilar. Let
    :math:`N` and :math:`K` denote mini-batch size and the dimension of input
    variables, respectively. The shape of both input variables should be
    ``(N, K)``.

    .. math::
        L = \\frac{1}{2N} \\left( \\sum_{n=1}^N y_n d_n^2
            + (1 - y_n) \\max ({\\rm margin} - d_n, 0)^2 \\right)

    where :math:`d_n = \\| {\\bf x_0}_n - {\\bf x_1}_n \\|_2`. :math:`N`
    denotes the mini-batch size. Input variables, x0 and x1, have :math:`N`
    vectors, and each vector is K-dimensional. Therefore, :math:`{\\bf x_0}_n`
    and :math:`{\\bf x_1}_n` are :math:`n`-th K-dimensional vectors of x0 and
    x1.

    Args:
        x0 (~chainer.Variable): The first input variable. The shape should be
            (N, K), where N denotes the mini-batch size, and K denotes the
            dimension of ``x0``.
        x1 (~chainer.Variable): The second input variable. The shape should be
            the same as ``x0``.
        y (~chainer.Variable): Labels. All values should be 0 or 1. The shape
            should be ``(N,)``, where N denotes the mini-batch size.
        margin (float): A parameter for contrastive loss. It should be positive
            value.

    Returns:
        ~chainer.Variable: A variable holding a scalar that is the loss value
            calculated by the above equation.

    .. note::
        This cost can be used to train siamese networks. See `Learning a
        Similarity Metric Discriminatively, with Application to Face
        Verification <http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf>`_
        for details.

    """
    return Contrastive(margin)(x0, x1, y)
