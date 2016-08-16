from chainer.functions.connection import deconvolution_nd
from chainer import initializers
from chainer import link
from chainer.utils import conv_nd


class DeconvolutionND(link.Link):
    """
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
