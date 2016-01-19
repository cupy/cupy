import numpy

from chainer import cuda
from chainer.functions.activation import maxout
from chainer import link
from chainer.links.connection import linear


class Maxout(link.Chain):
    """Maxout Networks

    It has three dimensional weight tensor whose shape is ``(M, C, N)``
    and optional bias vector whose shape is ``(M, C)`` where
   ``M`` is an output dimension,``C`` the number of channel, and
    ``N`` an input dimension . It computes

    .. math::

      Y_{i} = \\mathrm{max}_{j} (W_{ij\\cdot}x + b_{ij}).

    Here, :math:`x` is a input vector and :math:`W_{ij\\cdot}`
    is a sub-vector extracted from :math:`W` by fixing first
    and second dimensions to :math:`i` and :math:`j`, respectively.
    Minibatch dimension is omitted in the above equation.

    As an actual implementation, this Chain has a linear Link with
    a ``(M * C, N)`` weight vector and an optional ``M * C``
    dimensional bias.

    Args:
        in_size (int): Dimension of input vectors.
        out_size (int): Dimension of output vectors.
        num_channel (int): Number of channels.
        wscale (float): Scaling factor of the weight matrix.
        initialW (3-D array or None): Initial weight value.
            If ``None``, then this function uses ``wscale`` to initialize.
        initial_bias (2-D array, float or None): Initial bias value.
            If it is float, initial bias is filled with this value.
            If it is ``None``, bias is omitted.

    Attributes:
        linear (~chainer.Link): The Linear Link that performs
        affine transformation.

    .. seealso:: :func:`~chainer.functions.maxout`

    .. seealso::
         Goodfellow, I., Warde-farley, D., Mirza, M.,
         Courville, A., & Bengio, Y. (2013).
         Maxout Networks. In Proceedings of the 30th International
         Conference on Machine Learning (ICML-13) (pp. 1319-1327).
         `URL <http://jmlr.org/proceedings/papers/v28/goodfellow13.html>`_
    """

    def __init__(self, in_size, out_size, num_channel,
                 wscale=1, initialW=None, initial_bias=0):
        linear_out_size = out_size * num_channel
        if initialW is not None:
            initialW = initialW.reshape(linear_out_size, in_size)

        if initial_bias is not None:
            if numpy.isscalar(initial_bias):
                initial_bias = numpy.full(
                    (linear_out_size,), initial_bias, dtype=numpy.float32)
            elif isinstance(initial_bias, (numpy.ndarray, cuda.ndarray)):
                initial_bias = initial_bias.reshape(linear_out_size)
            else:
                raise ValueError(
                    'initial bias must be float, ndarray, or None')

        super(Maxout, self).__init__(
            linear=linear.Linear(
                in_size, linear_out_size, wscale,
                nobias=initial_bias is None, initialW=initialW,
                initial_bias=initial_bias))
        self.out_size = out_size
        self.num_channel = num_channel

    def __call__(self, x):
        """Applies the maxout layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the maxout layer.
        """
        y = self.linear(x)
        return maxout.maxout(y, self.num_channel)
