import numpy

from chainer.functions.connection import convolution_2d
from chainer import link


class Convolution2D(link.Link):

    """Two-dimensional convolutional layer.

    This link wraps the :func:`~chainer.functions.convolution_2d` function and
    holds the filter weight and bias vector as parameters.

    Args:
        in_channels (int): Number of channels of input arrays.
        out_channels (int): Number of channels of output arrays.
        ksize (int or pair of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        wscale (float): Scaling factor of the initial weight.
        bias (float): Initial bias value.
        nobias (bool): If ``True``, then this link does not use the bias term.
        use_cudnn (bool): If ``True``, then this link uses cuDNN if available.
        initialW (4-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.

    .. seealso::
       See :func:`chainer.functions.convolution_2d` for the definition of
       two-dimensional convolution.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    """
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False, use_cudnn=True,
                 initialW=None, initial_bias=None):
        kh, kw = _pair(ksize)
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.use_cudnn = use_cudnn

        W_shape = (out_channels, in_channels, kh, kw)
        super(Convolution2D, self).__init__(W=W_shape)

        if initialW is not None:
            self.W.data[...] = initialW
        else:
            std = wscale * numpy.sqrt(1. / (kh * kw * in_channels))
            self.W.data[...] = numpy.random.normal(0, std, W_shape)

        if nobias:
            self.b = None
        else:
            self.add_param('b', out_channels)
            if initial_bias is None:
                initial_bias = bias
            self.b.data[...] = initial_bias

    def __call__(self, x):
        """Applies the convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.

        """
        return convolution_2d.convolution_2d(
            x, self.W, self.b, self.stride, self.pad, self.use_cudnn)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
