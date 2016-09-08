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
            initializer instance or another value that
            :func:`~chainer.init_weight` helper function can take. This link
            uses :func:`~chainer.init_weight` to initialize the filter weight
            and passes the value of ``initialW`` to it as it is.
        initial_bias: Value used to initialize the bias vector. May be an
            initializer instance or another value except ``None`` that
            :func:`~chainer.init_weight` helper function can take. If ``None``
            is given, this link does not use the bias vector. This link uses
            :func:`~chainer.init_weight` to initialize the bias vector and
            passes the value of ``initial_bias`` to it as it is.
        use_cudnn (bool): If ``True``, then this link uses cuDNN if available.

    .. seealso::
       :func:`~chainer.functions.deconvolution_nd`

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter. If ``initial_bias`` is ``None``,
            set to ``None``.

    """

    def __init__(self, ndim, in_channels, out_channels, ksize, stride=1, pad=0,
                 outsize=None, initialW=None, initial_bias=None,
                 use_cudnn=True):
        ksize = conv_nd.as_tuple(ksize, ndim)
        self.stride = stride
        self.pad = pad
        self.use_cudnn = use_cudnn
        self.outsize = outsize

        W_shape = (in_channels, out_channels) + ksize
        super(DeconvolutionND, self).__init__(W=W_shape)
        initializers.init_weight(self.W.data, initialW)

        if initial_bias is None:
            self.b = None
        else:
            self.add_param('b', out_channels)
            initializers.init_weight(self.b.data, initial_bias)

    def __call__(self, x):
        return deconvolution_nd.deconvolution_nd(
            x, self.W, b=self.b, stride=self.stride, pad=self.pad,
            outsize=self.outsize, use_cudnn=self.use_cudnn)
