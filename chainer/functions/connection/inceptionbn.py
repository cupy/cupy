from chainer import function
from chainer import function_set
from chainer.functions.activation import relu
from chainer.functions.array import concat
from chainer.functions.connection import convolution_2d
from chainer.functions.normalization import batch_normalization
from chainer.functions.pooling import average_pooling_2d
from chainer.functions.pooling import max_pooling_2d


class InceptionBN(function.Function):
    """Inception module of the new GoogLeNet with BatchNormalization.

    This class acts like :class:`Inception`, while InceptionBN uses the
    :class:`BatchNormalization` on top of each convolution, the 5x5 convolution
    path is replaced by two consecutive 3x3 convolution applications, and the
    pooling method is configurable.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing \
    Internal Covariate Shift <http://arxiv.org/abs/1502.03167>`_.

    Args:
        in_channels (int): Number of channels of input arrays.
        out1 (int): Output size of the 1x1 convolution path.
        proj3 (int): Projection size of the single 3x3 convolution path.
        out3 (int): Output size of the single 3x3 convolution path.
        proj33 (int): Projection size of the double 3x3 convolutions path.
        out33 (int): Output size of the double 3x3 convolutions path.
        pooltype (str): Pooling type. It must be either ``'max'`` or ``'avg'``.
        proj_pool (bool): If True, do projection in the pooling path.
        stride (int): Stride parameter of the last convolution of each path.

    .. seealso:: :class:`Inception`

    """
    def __init__(self, in_channels, out1, proj3, out3, proj33, out33,
                 pooltype, proj_pool=None, stride=1):
        if out1 > 0:
            assert stride == 1
            assert proj_pool is not None

        self.f = function_set.FunctionSet(
            proj3=convolution_2d.Convolution2D(
                in_channels, proj3, 1, nobias=True),
            conv3=convolution_2d.Convolution2D(
                proj3, out3, 3, pad=1, stride=stride, nobias=True),
            proj33=convolution_2d.Convolution2D(
                in_channels, proj33, 1, nobias=True),
            conv33a=convolution_2d.Convolution2D(
                proj33, out33, 3, pad=1, nobias=True),
            conv33b=convolution_2d.Convolution2D(
                out33, out33, 3, pad=1, stride=stride, nobias=True),
            proj3n=batch_normalization.BatchNormalization(proj3),
            conv3n=batch_normalization.BatchNormalization(out3),
            proj33n=batch_normalization.BatchNormalization(proj33),
            conv33an=batch_normalization.BatchNormalization(out33),
            conv33bn=batch_normalization.BatchNormalization(out33),
        )

        if out1 > 0:
            self.f.conv1 = convolution_2d.Convolution2D(
                in_channels, out1, 1, stride=stride, nobias=True)
            self.f.conv1n = batch_normalization.BatchNormalization(out1)
        self.out1 = out1

        if proj_pool is not None:
            self.f.poolp = convolution_2d.Convolution2D(
                in_channels, proj_pool, 1, nobias=True)
            self.f.poolpn = batch_normalization.BatchNormalization(proj_pool)
        self.proj_pool = proj_pool

        if pooltype == 'max':
            self.f.pool = max_pooling_2d.MaxPooling2D(3, stride=stride, pad=1)
        elif pooltype == 'avg':
            self.f.pool = average_pooling_2d.AveragePooling2D(
                3, stride=stride, pad=1)
        else:
            raise NotImplementedError()

    def __call__(self, x):
        outs = []

        if self.out1 > 0:
            h1 = self.f.conv1(x)
            h1 = self.f.conv1n(h1)
            h1 = relu.relu(h1)
            outs.append(h1)

        h3 = relu.relu(self.f.proj3n(self.f.proj3(x)))
        h3 = relu.relu(self.f.conv3n(self.f.conv3(h3)))
        outs.append(h3)

        h33 = relu.relu(self.f.proj33n(self.f.proj33(x)))
        h33 = relu.relu(self.f.conv33an(self.f.conv33a(h33)))
        h33 = relu.relu(self.f.conv33bn(self.f.conv33b(h33)))
        outs.append(h33)

        p = self.f.pool(x)
        if self.proj_pool is not None:
            p = relu.relu(self.f.poolpn(self.f.poolp(p)))
        outs.append(p)

        y = concat.concat(outs, axis=1)
        return y

    # forward of InceptionBN is a series of forward operations
    # of existing functions. So we do not need additional type checks.
    def check_type_forward(self, in_types):
        pass

    def to_gpu(self, device=None):
        super(InceptionBN, self).to_gpu(device)
        self.f.to_gpu(device)

    @property
    def parameters(self):
        return self.f.parameters

    @parameters.setter
    def parameters(self, params):
        self.f.parameters = params

    @property
    def gradients(self):
        return self.f.gradients

    @gradients.setter
    def gradients(self, grads):
        self.f.gradients = grads
