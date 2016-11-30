from chainer.functions.connection import convolution_nd
from chainer import initializers
from chainer import link
from chainer.utils import conv_nd


class ConvolutionND(link.Link):
    """N-dimensional convolution layer.

    This link wraps the :func:`~chainer.functions.convolution_nd` function and
    holds the filter weight and bias vector as parameters.

    Args:
        ndim (int): Number of spatial dimensions.
        in_channels (int): Number of channels of input arrays.
        out_channels (int): Number of channels of output arrays.
        ksize (int or tuple of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k, ..., k)`` are equivalent.
        stride (int or tuple of ints): Stride of filter application.
            ``stride=s`` and ``stride=(s, s, ..., s)`` are equivalent.
        pad (int or tuple of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.
        initialW: Value used to initialize the filter weight. May be an
            initializer instance or another value that
            :func:`~chainer.init_weight` helper function can take.
        initial_bias: Value used to initialize the bias vector. May be an
            initializer instance or another value except ``None`` that
            :func:`~chainer.init_weight` helper function can take. If ``None``
            is given, this link does not use the bias vector.
        use_cudnn (bool): If ``True``, then this link uses cuDNN if available.
            See :func:`~chainer.functions.convolution_nd` for exact conditions
            of cuDNN availability.
        cover_all (bool): If ``True``, all spatial locations are convoluted
            into some output pixels. It may make the output size larger.
            ``cover_all`` needs to be ``False`` if you want to use cuDNN.

    .. seealso::
        See :func:`~chainer.functions.convolution_nd` for the definition of
        N-dimensional convolution. See
        :func:`~chainer.functions.convolution_2d` for the definition of
        two-dimensional convolution.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter. If ``initial_bias`` is ``None``,
            set to ``None``.

    """

    def __init__(self, ndim, in_channels, out_channels, ksize, stride=1, pad=0,
                 initialW=None, initial_bias=None, use_cudnn=True,
                 cover_all=False):
        ksize = conv_nd.as_tuple(ksize, ndim)
        self.stride = stride
        self.pad = pad
        self.use_cudnn = use_cudnn
        self.cover_all = cover_all

        super(ConvolutionND, self).__init__()

        W_shape = (out_channels, in_channels) + ksize
        initialW = initializers._get_initializer(initialW)
        self.add_param('W', W_shape, initializer=initialW)

        if initial_bias is None:
            self.b = None
        else:
            initial_bias = initializers._get_initializer(initial_bias)
            self.add_param('b', out_channels, initializer=initial_bias)

    def __call__(self, x):
        """Applies N-dimensional convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of convolution.

        """
        return convolution_nd.convolution_nd(
            x, self.W, self.b, self.stride, self.pad,
            use_cudnn=self.use_cudnn, cover_all=self.cover_all)
