import math

from chainer import cuda
from chainer.functions.connection import depthwise_convolution_2d
from chainer import initializers
from chainer import link


class DepthwiseConvolution2D(link.Link):

    """Two-dimensional depthwise convolutional layer.

    This link wraps the :func:`~chainer.functions.depthwise_convolution_2d`
    function and holds the filter weight and bias vector as parameters.

    Args:
        in_channels (int): Number of channels of input arrays. If ``None``,
            parameter initialization will be deferred until the first forward
            data pass at which time the size will be determined.
        channel_multiplier (int): Channel multiplier number. Number of output
            arrays equal ``in_channels * channel_multiplier``.
        ksize (int or pair of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        wscale (float): Scaling factor of the initial weight.
        bias (float): Initial bias value.
        nobias (bool): If ``True``, then this link does not use the bias term.
        initialW (4-D array): Initial weight value. If ``None``, then this
            function uses scaled Gaussian distribution to initialize weight.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses ``bias`` to initialize bias.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.

    .. seealso::
       See :func:`chainer.functions.depthwise_convolution_2d`.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    """

    def __init__(self, in_channels, channel_multiplier, ksize, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False, initialW=None,
                 initial_bias=None):
        super(DepthwiseConvolution2D, self).__init__()
        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.channel_multiplier = channel_multiplier
        self.nobias = nobias

        # For compatibility with Convolution2D
        self.initialW = initialW
        self.wscale = wscale

        # For compatibility with Convolution2D, the scale of weights is
        # proportional to the square root of wscale.
        self._W_initializer = initializers._get_initializer(
            initialW, scale=math.sqrt(wscale))

        if nobias:
            self.b = None
        else:
            if initial_bias is None:
                initial_bias = bias
            self.bias_initilizer = initializers._get_initializer(initial_bias)
            if in_channels is None:
                self.add_uninitialized_param('b')

        if in_channels is None:
            self.add_uninitialized_param('W')
        else:
            self._initialize_params(in_channels)

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        W_shape = (self.channel_multiplier, in_channels, kh, kw)
        self.add_param('W', W_shape, initializer=self._W_initializer)
        if not self.nobias:
            self.add_param('b', self.channel_multiplier * in_channels,
                           initializer=self.bias_initilizer)

    def __call__(self, x):
        """Applies the depthwise convolution layer.

        Args:
            x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
                Input image.

        Returns:
            ~chainer.Variable: Output of the depthwise convolution.

        """
        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params(x.shape[1])
        return depthwise_convolution_2d.depthwise_convolution_2d(
            x, self.W, self.b, self.stride, self.pad)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
