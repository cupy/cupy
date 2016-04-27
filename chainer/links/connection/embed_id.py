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
        ignore_label (int or None): If ``ignore_label`` is an int value,
            ``i``-th column of return value is filled with ``0``.

    .. seealso:: :func:`chainer.functions.embed_id`

    Attributes:
        W (~chainer.Variable): Embedding parameter matrix.

    """

    ignore_label = None

    def __init__(self, in_size, out_size, ignore_label=None):
        super(EmbedID, self).__init__(W=(in_size, out_size))
        self.W.data[...] = numpy.random.randn(in_size, out_size)
        self.ignore_label = ignore_label

    def __call__(self, x):
        """Extracts the word embedding of given IDs.

        Args:
            x (~chainer.Variable): Batch vectors of IDs.

        Returns:
            ~chainer.Variable: Batch of corresponding embeddings.

        """
        return embed_id.embed_id(x, self.W, ignore_label=self.ignore_label)
