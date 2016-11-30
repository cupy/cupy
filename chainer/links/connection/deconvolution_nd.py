from chainer.functions.connection import deconvolution_nd
from chainer import initializers
from chainer import link
from chainer.utils import conv_nd


class DeconvolutionND(link.Link):
    """N-dimensional deconvolution function.

    This link wraps :func:`~chainer.functions.deconvolution_nd` function and
    holds the filter weight and bias vector as its parameters.

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
        outsize (tuple of ints): Expected output size of deconvolutional
            operation. It should be a tuple of ints that represents the output
            size of each dimension. Default value is ``None`` and the outsize
            is estimated with input size, stride and pad.
        initialW: Value used to initialize the filter weight. May be an
            initializer instance of another value the same with that
            :func:`~chainer.init_weight` function can take.
        initial_bias: Value used to initialize the bias vector. May be an
            initializer instance or another value except ``None`` the same with
            that :func:`~chainer.init_weight` function can take. If ``None`` is
            supplied, this link does not use the bias vector.
        use_cudnn (bool): If ``True``, then this link uses cuDNN if available.

    .. seealso::
       :func:`~chainer.functions.deconvolution_nd`

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter. If ``initial_bias`` is ``None``,
            set to ``None``.

    """

    def __init__(self, ndim, in_channels, out_channels, ksize, stride=1, pad=0,
                 outsize=None, initialW=None, initial_bias=0, use_cudnn=True):
        ksize = conv_nd.as_tuple(ksize, ndim)
        self.stride = stride
        self.pad = pad
        self.use_cudnn = use_cudnn
        self.outsize = outsize

        super(DeconvolutionND, self).__init__()

        W_shape = (in_channels, out_channels) + ksize
        initialW = initializers._get_initializer(initialW)
        self.add_param('W', W_shape, initializer=initialW)

        if initial_bias is None:
            self.b = None
        else:
            initial_bias = initializers._get_initializer(initial_bias)
            self.add_param('b', out_channels, initializer=initial_bias)

    def __call__(self, x):
        return deconvolution_nd.deconvolution_nd(
            x, self.W, b=self.b, stride=self.stride, pad=self.pad,
            outsize=self.outsize, use_cudnn=self.use_cudnn)
