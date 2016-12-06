from chainer import initializers
from chainer import link

from chainer.functions.array import broadcast
from chainer.functions.math import bias
from chainer.functions.math import scale
from chainer.functions.math import sqrt
from chainer.functions.math import square
from chainer.functions.math import sum


class LayerNormalization(link.Chain):

    """Layer normalization layer on outputs of linear functions.

    This link implements a "layer normalization" layer
    which normalizes the input units by statistics
    that are computed along the second axis,
    scales and shifts them.


    Args:
        size (int): Size of input units.
        eps (float): Epsilon value for numerical stability of normalization.
        initial_gamma (~chainer.Initializer): Initializer for scaling vector.
            If ``None``, then the vector is initialized
            by :class:`~chainer.initializers.HeNormal`.
            If a scalar, the vectors are filled by it.
            If ``numpy.ndarray``, the vectors are set by it.
        initial_beta (~chainer.Initializer): Initializer for shifting vector.
            If ``None``, then the vector is initialized
            by :class:`~chainer.initializers.HeNormal`.
            If a scalar, the vectors are filled by it.
            If ``numpy.ndarray``, the vectors are set by it.

    Attributes:
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        eps (float): Epsilon value for numerical stability.

    See: `Layer Normalization <https://arxiv.org/abs/1607.06450>`_
    """

    def __init__(self, size, eps=1e-6, initial_gamma=None, initial_beta=None):
        super(LayerNormalization, self).__init__()
        self.add_param('gamma', size)
        if initial_gamma is None:
            initial_gamma = initializers.One()
        initializers.init_weight(self.gamma.data, initial_gamma)

        self.add_param('beta', size)
        if initial_beta is None:
            initial_beta = initializers.Zero()
        initializers.init_weight(self.beta.data, initial_beta)
        self.eps = eps

    def _normalize(self, x):
        size = x.shape[1]
        mean = broadcast.broadcast_to(
            (sum.sum(x, axis=1) / size)[:, None],
            x.shape)
        std = broadcast.broadcast_to(sqrt.sqrt(
            sum.sum(square.square(x - mean), axis=1) / size)[:, None],
            x.shape) + self.eps
        return (x - mean) / std

    def __call__(self, x):
        """Apply layer normalization to given input.

        Args:
            x (~chainer.Variable): Batch vectors.
                Shape of this value must be `(batch_size, unit_size)`,
                e.g., the output of :func:`~chainer.functions.linear`.

        Returns:
            ~chainer.Variable: Output of the layer normalization.

        """
        normalized = self._normalize(x)
        return bias.bias(scale.scale(normalized, self.gamma), self.beta)
