import math

from chainer.functions.connection import convolution_nd
from chainer import initializers
from chainer import link


class ConvolutionND(link.Link):
    """N-dimensional convolution layer.

    This link wraps the :func:`~chainer.functions.convolution_nd` function and
    holds the filter weight and bias vector as parameters.

    Args:
        N (int): Number of spacial dimensions.
        in_channels (int): Number of channels of input arrays.
        out_channels (int): Number of channels of output arrays.
        ksize (int or tuple of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k, ..., k)`` are equivalent.
        stride (int or tuple of ints): Stride of filter application.
            ``stride=s`` and ``stride=(s, s, ..., s)`` are equivalent.
        pad (int or tuple of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.
        wscale (float): Scaling factor of the initial weight.
        bias (float): Initial bias value.
        nobias (bool): If ``True``, then this link does not use the bias term.
        use_cudnn (bool): If ``True``, then this link uses cuDNN if available.
        initialW (N-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.

    .. seealso::
        See :func:`chainer.functions.convolution_nd` for the definition of
        N-dimensional convolution.
        See :func:`chainer.functions.convolution_2d` for the definition of
        two-dimensional convolution.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    """

    def __init__(self, N, in_channels, out_channels, ksize, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False, use_cudnn=True,
                 initialW=None, initial_bias=None):
        ks = _ensure_tuple(ksize, N)
        self.stride = _ensure_tuple(stride, N)
        self.pad = _ensure_tuple(pad, N)
        self.use_cudnn = use_cudnn

        W_shape = (out_channels, in_channels) + ks
        super(ConvolutionND, self).__init__(W=W_shape)

        # For backward compatibility, the scale of weights is proportional to
        # the Nth root of wscale.
        # TODO(takagi) Nth root right?
        initializers.init_weight(self.W.data, initialW,
                                 scale=math.pow(wscale, 1.0 / N))

        if nobias:
            self.b = None
        else:
            self.add_param('b', out_channels)
            if initial_bias is None:
                initial_bias = bias
            initializers.init_weight(self.b.data, initial_bias)

    def __call__(self, x):
        """Applies N-dimensional convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of convolution.

        """
        return convolution_nd.convolution_nd(
            x, self.W, self.b, self.stride, self.pad, self.use_cudnn)


def _ensure_tuple(x, n):
    if hasattr(x, '__getitem__'):
        return x
    return tuple([x] * n)
