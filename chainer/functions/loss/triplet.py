import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Triplet(function.Function):

    """Triplet loss function."""

    def __init__(self, margin):
        if margin <= 0:
            raise ValueError('margin should be positive value.')
        self.margin = margin

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)

        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[2].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape,
            in_types[0].shape == in_types[2].shape,
            in_types[0].shape[0] > 0
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)

        anchor, positive, negative = inputs
        N = anchor.shape[0]

        dist = xp.sum(
            (anchor - positive) ** 2 - (anchor - negative) ** 2,
            axis=1) + self.margin
        self.dist_hinge = xp.maximum(dist, 0)
        loss = xp.sum(self.dist_hinge) / N

        return xp.array(loss, dtype=numpy.float32),

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)

        anchor, positive, negative = inputs
        N = anchor.shape[0]

        x_dim = anchor.shape[1]
        tmp = xp.repeat(self.dist_hinge[:, None], x_dim, axis=1)
        mask = xp.array(tmp > 0, dtype=numpy.float32)

        tmp = 2 * gy[0] * mask / N
        gx0 = (tmp * (negative - positive)).astype(numpy.float32)
        gx1 = (tmp * (positive - anchor)).astype(numpy.float32)
        gx2 = (tmp * (anchor - negative)).astype(numpy.float32)

        return gx0, gx1, gx2


def triplet(anchor, positive, negative, margin=0.2):
    """Computes triplet loss.

    It takes a triplet of variables as inputs, :math:`a`, :math:`p` and
    :math:`n`: anchor, positive example and negative example respectively.
    The triplet defines a relative similarity between samples.
    Let :math:`N` and :math:`K` denote mini-batch size and the dimension of
    input variables, respectively. The shape of all input variables should be
    :math:`(N, K)`.

    .. math::
        L(a, p, n) = \\frac{1}{N} \\left( \\sum_{i=1}^N \\max \{d(a_i, p_i)
            - d(a_i, n_i) + {\\rm margin}, 0\} \\right)

    where :math:`d(x_i, y_i) = \\| {\\bf x}_i - {\\bf y}_i \\|_2^2`.

    Args:
        anchor (~chainer.Variable): The anchor example variable. The shape
            should be :math:`(N, K)`, where :math:`N` denotes the minibatch
            size, and :math:`K` denotes the dimension of the anchor.
        positive (~chainer.Variable): The positive example variable. The shape
            should be the same as anchor.
        negative (~chainer.Variable): The negative example variable. The shape
            should be the same as anchor.
        margin (float): A parameter for triplet loss. It should be a positive
            value.

    Returns:
        ~chainer.Variable: A variable holding a scalar that is the loss value
            calculated by the above equation.

    .. note::
        This cost can be used to train triplet networks. See `Learning \
        Fine-grained Image Similarity with Deep Ranking \
        <https://arxiv.org/abs/1404.4661>`_ for details.
    """
    return Triplet(margin)(anchor, positive, negative)
