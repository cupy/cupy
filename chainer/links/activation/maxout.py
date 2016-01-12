import numpy

from chainer.functions.activation import maxout
from chainer import link


class Maxout(link.Link):
    """Maxout Networks

    Args:
        in_size (int): Dimension of input vectors.
        num_channel (int): Number of channels.
        out_size (int): Dimension of output vectors.
        wscale (float): Scaling factor of the weight matrix.
        initialW (3-D array): Initial weight value. If ``None``, then this
            function uses ``wscale`` to initialize.
        initial_bias (2-D array): Initial bias value. If ``None``, then this
            functions uses ``bias`` to initialize.

    Attributes:
        W (~chainer.Variable): Weight tensor with shape
            ``(in_size, num_channel, out_size)``.
        b (~chainer.Variable): Bias vector with shape
            ``(num_channel, out_size)``.

    .. seealso:: :func:`~chainer.functions.maxout`

    .. seealso::
         Goodfellow, I., Warde-farley, D., Mirza, M.,
         Courville, A., & Bengio, Y. (2013).
         Maxout Networks. In Proceedings of the 30th International
         Conference on Machine Learning (ICML-13) (pp. 1319-1327).
         `URL <http://jmlr.org/proceedings/papers/v28/goodfellow13.html>`_
    """

    def __init__(self, in_size, num_channel, out_size,
                 wscale=1, initialW=None, initial_bias=0):
        super(Maxout, self).__init__(W=(in_size, num_channel, out_size))
        if initialW is None:
            initialW = numpy.random.normal(
                0, wscale * numpy.sqrt(1. / in_size), self.W.data.shape)
        self.W.data[...] = initialW

        if initial_bias is not None:
            self.add_param('b', (num_channel, out_size))
            self.b.data[...] = initial_bias
        else:
            self.b = None

    def __call__(self, x):
        """Applies the maxout layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the maxout layer.
        """

        return maxout.maxout(x, self.W, self.b)
