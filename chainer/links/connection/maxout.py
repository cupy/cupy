import numpy

from chainer.functions.connections import maxout
from chainer import link


class Maxout(link.Link):
    """Maxout layer

    description here
    Args:
        in_size (int): Dimension of input vectors.
        num_channel (int): Number of channels.
        out_size (int): Dimension of output vectors.
        wscale (float): Scaling factor of the weight matrix.
        initialW (3-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
        initial_bias (2-D array): Initial bias value. If ``None``, then this
            functions uses to initialize ``bias``.

    .. seealso:: :func:`~chainer.functions.maxout`

    Attributes:
        W (~chainer.Variable): w
        b (~chainer.Variable): w
    """

    def __init__(self, in_size, num_channel, out_size,
                 wscale, initialW, initial_bias):
        super(Maxout, self).__init__(W=(in_size, num_channel, out_size))
        if initialW is None:
            initialW = numpy.random.normal(
                0, wscale * numpy.sqrt(1. / in_size), (out_size, in_size))
        self.W.data[...] = initialW

        if initial_bias is not None:
            self.add_param('b', (num_channel, out_size))
            self.b.data[...] = initial_bias

    def __call__(self, x):
        """Applies the maxout layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the maxout layer.
        """

        return maxout.maxout(x, self.W, self.b)
