from chainer.functions.connection import embed_id
from chainer import initializations
from chainer import link


class EmbedID(link.Link):

    """Efficient linear layer for one-hot input.

    This is a link that wraps the :func:`~chainer.functions.embed_id` function.
    This link holds the ID (word) embedding matrix ``W`` as a parameter.

    Args:
        in_size (int): Number of different identifiers (a.k.a. vocabulary
            size).
        out_size (int): Size of embedding vector.
        initialW (2-D array): Initial weight value. If ``None``, then the
                matrix is initialized from the standard normal distribution.
            May also be a callable that takes a tuple that represents
            the shape of the matrix and returns a matrix of the same
            dimensions to use for initialization.

    .. seealso:: :func:`chainer.functions.embed_id`

    Attributes:
        W (~chainer.Variable): Embedding parameter matrix.

    """

    def __init__(self, in_size, out_size, initialW=None):
        super(EmbedID, self).__init__(W=(in_size, out_size))
        initializations.init_weight(
            self.W.data, initialW,
            none_default=lambda s: initializations.normal(s, scale=1.0))

    def __call__(self, x):
        """Extracts the word embedding of given IDs.

        Args:
            x (~chainer.Variable): Batch vectors of IDs.

        Returns:
            ~chainer.Variable: Batch of corresponding embeddings.

        """
        return embed_id.embed_id(x, self.W)
