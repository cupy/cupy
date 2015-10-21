import numpy

from chainer.functions.connection import embed_id
from chainer import link


class EmbedID(link.Link):

    """Efficient linear layer for one-hot input.

    This is a link that wraps the :func:`~chainer.functions.embed_id` function.
    This link holds the ID (word) embedding matrix ``W`` as a parameter.

    Args:
        in_size (int): Number of different identifiers (a.k.a. vocabulary
            size).
        out_size (int): Size of embedding vector.

    Attributes:
        W (~chainer.Variable): Embedding parameter matrix.

    .. seealso:: :func:`chainer.functions.embed_id`

    """
    def __init__(self, in_size, out_size):
        super(EmbedID, self).__init__(W=(in_size, out_size))
        self.W.data[...] = numpy.random.randn(in_size, out_size)

    def __call__(self, x):
        """Extracts the word embedding of given IDs.

        Args:
            x (~chainer.Variable): Batch vectors of IDs.

        Returns:
            ~chainer.Variable: Batch of corresponding embeddings.

        """
        return embed_id.embed_id(x, self.W)
