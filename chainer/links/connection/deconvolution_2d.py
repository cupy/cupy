import math

import numpy

from chainer.functions.connection import deconvolution_2d
from chainer import link


class Deconvolution2D(link.Link):

    """Two dimensional deconvolution function.

    This link wraps the :func:`~chainer.functions.deconvolution_2d` function
    and holds the filter weight and bias vector as parameters.

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
        nobias (bool): If ``True``, then this function does not use the bias
            term.
        outsize (tuple): Expected output size of deconvolutional operation.
            It should be pair of height and width :math:`(out_H, out_W)`.
            Default value is ``None`` and the outsize is estimated by
            input size, stride and pad.
        use_cudnn (bool): If ``True``, then this function uses cuDNN if
            available.
        initialW (4-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.

    The filter weight has four dimensions :math:`(c_I, c_O, k_H, k_W)`
    which indicate the number of the number of input channels, output channels,
    height and width of the kernels, respectively.
    The filter weight is initialized with i.i.d. Gaussian random samples, each
    of which has zero mean and deviation :math:`\\sqrt{1/(c_I k_H k_W)}` by
    default. The deviation is scaled by ``wscale`` if specified.

    The bias vector is of size :math:`c_O`.
    Its elements are initialized by ``bias`` argument.
    If ``nobias`` argument is set to True, then this function does not hold
    the bias parameter.

    .. seealso::
       See :func:`chainer.functions.deconvolution_2d` for the definition of
       two-dimensional convolution.
    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False, outsize=None, use_cudnn=True,
                 initialW=None, initial_bias=None):
        kh, kw = _pair(ksize)
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.outsize = (None, None) if outsize is None else outsize
        self.use_cudnn = use_cudnn

        W_shape = (in_channels, out_channels, kh, kw)
        super(Deconvolution2D, self).__init__(W=W_shape)

        if initialW is not None:
            assert initialW.shape == \
                (in_channels, out_channels, kh, kw)
            self.W.data[...] = initialW
        else:
            self.W.data = numpy.random.normal(
                0, wscale * math.sqrt(1. / (kh * kw * in_channels)),
                (in_channels, out_channels, kh, kw)
            ).astype(numpy.float32)

        if initial_bias is not None:
            assert initial_bias.shape == (out_channels,)
            self.add_param('b', out_channels)
            self.b.data[...] = initial_bias
        elif not nobias:
            self.add_param('b', out_channels)
            self.b.data = numpy.repeat(numpy.float32(bias), out_channels)
        else:
            self.b = None

    def __call__(self, x):
        return deconvolution_2d.deconvolution_2d(
            x, self.W, self.b, self.stride, self.pad,
            self.outsize, self.use_cudnn)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
