from chainer.functions.activation import relu
from chainer.functions.array import concat
from chainer.functions.pooling import max_pooling_2d
from chainer import link
from chainer.links.connection import convolution_2d


class Inception(link.Chain):

    """Inception module of GoogLeNet.

    It applies four different functions to the input array and concatenates
    their outputs along the channel dimension. Three of them are 2D
    convolutions of sizes 1x1, 3x3 and 5x5. Convolution paths of 3x3 and 5x5
    sizes have 1x1 convolutions (called projections) ahead of them. The other
    path consists of 1x1 convolution (projection) and 3x3 max pooling.

    The output array has the same spatial size as the input. In order to
    satisfy this, Inception module uses appropriate padding for each
    convolution and pooling.

    See: `Going Deeper with Convolutions <https://arxiv.org/abs/1409.4842>`_.

    Args:
        in_channels (int): Number of channels of input arrays.
        out1 (int): Output size of 1x1 convolution path.
        proj3 (int): Projection size of 3x3 convolution path.
        out3 (int): Output size of 3x3 convolution path.
        proj5 (int): Projection size of 5x5 convolution path.
        out5 (int): Output size of 5x5 convolution path.
        proj_pool (int): Projection size of max pooling path.
        conv_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the convolution matrix weights.
            Maybe be ``None`` to use default initialization.
        bias_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the convolution bias weights.
            Maybe be ``None`` to use default initialization.

    """

    def __init__(self, in_channels, out1, proj3, out3, proj5, out5, proj_pool,
                 conv_init=None, bias_init=None):
        super(Inception, self).__init__(
            conv1=convolution_2d.Convolution2D(in_channels, out1, 1,
                                               initialW=conv_init,
                                               initial_bias=bias_init),
            proj3=convolution_2d.Convolution2D(in_channels, proj3, 1,
                                               initialW=conv_init,
                                               initial_bias=bias_init),
            conv3=convolution_2d.Convolution2D(proj3, out3, 3, pad=1,
                                               initialW=conv_init,
                                               initial_bias=bias_init),
            proj5=convolution_2d.Convolution2D(in_channels, proj5, 1,
                                               initialW=conv_init,
                                               initial_bias=bias_init),
            conv5=convolution_2d.Convolution2D(proj5, out5, 5, pad=2,
                                               initialW=conv_init,
                                               initial_bias=bias_init),
            projp=convolution_2d.Convolution2D(in_channels, proj_pool, 1,
                                               initialW=conv_init,
                                               initial_bias=bias_init),
        )

    def __call__(self, x):
        """Computes the output of the Inception module.

        Args:
            x (~chainer.Variable): Input variable.

        Returns:
            Variable: Output variable. Its array has the same spatial size and
            the same minibatch size as the input array. The channel dimension
            has size ``out1 + out3 + out5 + proj_pool``.

        """
        out1 = self.conv1(x)
        out3 = self.conv3(relu.relu(self.proj3(x)))
        out5 = self.conv5(relu.relu(self.proj5(x)))
        pool = self.projp(max_pooling_2d.max_pooling_2d(
            x, 3, stride=1, pad=1))
        y = relu.relu(concat.concat((out1, out3, out5, pool), axis=1))
        return y
