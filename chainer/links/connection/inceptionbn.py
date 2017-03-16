import numpy

from chainer.functions.activation import relu
from chainer.functions.array import concat
from chainer.functions.pooling import average_pooling_2d
from chainer.functions.pooling import max_pooling_2d
from chainer import link
from chainer.links.connection import convolution_2d
from chainer.links.normalization import batch_normalization


class InceptionBN(link.Chain):

    """Inception module of the new GoogLeNet with BatchNormalization.

    This chain acts like :class:`Inception`, while InceptionBN uses the
    :class:`BatchNormalization` on top of each convolution, the 5x5 convolution
    path is replaced by two consecutive 3x3 convolution applications, and the
    pooling method is configurable.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing \
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_.

    Args:
        in_channels (int): Number of channels of input arrays.
        out1 (int): Output size of the 1x1 convolution path.
        proj3 (int): Projection size of the single 3x3 convolution path.
        out3 (int): Output size of the single 3x3 convolution path.
        proj33 (int): Projection size of the double 3x3 convolutions path.
        out33 (int): Output size of the double 3x3 convolutions path.
        pooltype (str): Pooling type. It must be either ``'max'`` or ``'avg'``.
        proj_pool (bool): If ``True``, do projection in the pooling path.
        stride (int): Stride parameter of the last convolution of each path.
        conv_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the convolution matrix weights.
            Maybe be ``None`` to use default initialization.
        dtype (numpy.dtype): Type to use in
            ``~batch_normalization.BatchNormalization``.

    .. seealso:: :class:`Inception`

    Attributes:
        train (bool): If ``True``, then batch normalization layers are used in
            training mode. If ``False``, they are used in testing mode.

    """

    def __init__(self, in_channels, out1, proj3, out3, proj33, out33,
                 pooltype, proj_pool=None, stride=1, conv_init=None,
                 dtype=numpy.float32):
        super(InceptionBN, self).__init__(
            proj3=convolution_2d.Convolution2D(
                in_channels, proj3, 1, nobias=True, initialW=conv_init),
            conv3=convolution_2d.Convolution2D(
                proj3, out3, 3, pad=1, stride=stride, nobias=True,
                initialW=conv_init),
            proj33=convolution_2d.Convolution2D(
                in_channels, proj33, 1, nobias=True, initialW=conv_init),
            conv33a=convolution_2d.Convolution2D(
                proj33, out33, 3, pad=1, nobias=True, initialW=conv_init),
            conv33b=convolution_2d.Convolution2D(
                out33, out33, 3, pad=1, stride=stride, nobias=True,
                initialW=conv_init),
            proj3n=batch_normalization.BatchNormalization(proj3, dtype=dtype),
            conv3n=batch_normalization.BatchNormalization(out3, dtype=dtype),
            proj33n=batch_normalization.BatchNormalization(proj33,
                                                           dtype=dtype),
            conv33an=batch_normalization.BatchNormalization(out33,
                                                            dtype=dtype),
            conv33bn=batch_normalization.BatchNormalization(out33,
                                                            dtype=dtype),
        )

        if out1 > 0:
            assert stride == 1
            assert proj_pool is not None
            self.add_link('conv1',
                          convolution_2d.Convolution2D(in_channels, out1, 1,
                                                       stride=stride,
                                                       nobias=True,
                                                       initialW=conv_init))
            self.add_link('conv1n', batch_normalization.BatchNormalization(
                out1, dtype=dtype))
        self.out1 = out1

        if proj_pool is not None:
            self.add_link('poolp', convolution_2d.Convolution2D(
                in_channels, proj_pool, 1, nobias=True, initialW=conv_init))
            self.add_link('poolpn', batch_normalization.BatchNormalization(
                proj_pool, dtype=dtype))
        self.proj_pool = proj_pool

        self.stride = stride
        self.pooltype = pooltype
        if pooltype != 'max' and pooltype != 'avg':
            raise NotImplementedError()

        self.train = True

    def __call__(self, x, test=None):
        """Computes the output of the InceptionBN module.

        Args:
            x (Variable): An input variable.
            test (bool): If ``True``, batch normalization layers run in testing
                mode; if ``test`` is omitted, ``not self.train`` is used as
                ``test``.

        """
        if test is None:
            test = not self.train
        outs = []

        if self.out1 > 0:
            h1 = self.conv1(x)
            h1 = self.conv1n(h1, test=test)
            h1 = relu.relu(h1)
            outs.append(h1)

        h3 = relu.relu(self.proj3n(self.proj3(x), test=test))
        h3 = relu.relu(self.conv3n(self.conv3(h3), test=test))
        outs.append(h3)

        h33 = relu.relu(self.proj33n(self.proj33(x), test=test))
        h33 = relu.relu(self.conv33an(self.conv33a(h33), test=test))
        h33 = relu.relu(self.conv33bn(self.conv33b(h33), test=test))
        outs.append(h33)

        if self.pooltype == 'max':
            p = max_pooling_2d.max_pooling_2d(x, 3, stride=self.stride, pad=1,
                                              cover_all=False)
        else:
            p = average_pooling_2d.average_pooling_2d(x, 3, stride=self.stride,
                                                      pad=1)
        if self.proj_pool is not None:
            p = relu.relu(self.poolpn(self.poolp(p), test=test))
        outs.append(p)

        y = concat.concat(outs, axis=1)
        return y
