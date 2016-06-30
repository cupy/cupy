import math

from chainer.functions.connection import dilated_convolution_2d
from chainer import initializers
from chainer import link


class DilatedConvolution2D(link.Link):

    """Two-dimensional dilated convolutional layer.
    """

    def __init__(self, in_channels, out_channels, ksize, dilate=1, stride=1, pad=0,
                 wscale=1, bias=0, nobias=False, use_cudnn=True,
                 initialW=None, initial_bias=None):
        kh, kw = _pair(ksize)
        self.dilate = _pair(dilate)
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.use_cudnn = use_cudnn

        W_shape = (out_channels, in_channels, kh, kw)
        super(DilatedConvolution2D, self).__init__(W=W_shape)

        # For backward compatibility, the scale of weights is proportional to
        # the square root of wscale.
        initializers.init_weight(self.W.data, initialW,
                                 scale=math.sqrt(wscale))

        if nobias:
            self.b = None
        else:
            self.add_param('b', out_channels)
            if initial_bias is None:
                initial_bias = bias
            initializers.init_weight(self.b.data, initial_bias)

    def __call__(self, x):
        """Applies the convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.

        """
        return dilated_convolution_2d.dilated_convolution_2d(
            x, self.W, self.b, self.dilate, self.stride, self.pad, self.use_cudnn)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
