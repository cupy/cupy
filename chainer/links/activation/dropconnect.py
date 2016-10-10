import math

from chainer import cuda
from chainer.functions.noise import dropconnect
from chainer import initializers
from chainer import link


class Dropconnect(link.Link):

    """Linear layer using dropconnect.

    Args:
        in_size (int): Dimension of input vectors. If None, parameter
            initialization will be deferred until the first forward data pass
            at which time the size will be determined.
        out_size (int): Dimension of output vectors.
        wscale (float): Scaling factor of the weight matrix.
        bias (float): Initial bias value.
        nobias (bool): If ``True``, then this function does not use the bias.
        initialW (2-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.

    .. seealso:: :func:`~chainer.functions.dropconnect`

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    """

    def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False,
                 ratio=.5, initialW=None, initial_bias=None):
        super(Dropconnect, self).__init__()

        # For backward compatibility
        self.initialW = initialW
        self.wscale = wscale

        self.out_size = out_size
        self.ratio = ratio
        # For backward compatibility, the scale of weights is proportional to
        # the square root of wscale.
        self._W_initializer = initializers._get_initializer(
            initialW, math.sqrt(wscale))

        if in_size is None:
            self.add_uninitialized_param('W')
        else:
            self._initialize_params(in_size)

        if nobias:
            self.b = None
        else:
            if initial_bias is None:
                initial_bias = bias
            bias_initializer = initializers._get_initializer(initial_bias)
            self.add_param('b', out_size, initializer=bias_initializer)

    def _initialize_params(self, in_size):
        self.add_param('W', (self.out_size, in_size),
                       initializer=self._W_initializer)

    def __call__(self, x, train=True):
        """Applies the dropconnect layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """
        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params(x.size // len(x.data))
        return dropconnect.dropconnect(x, self.W, self.b, self.ratio, train)
