import numpy
import six

from chainer import cuda
from chainer.functions.array import concat
from chainer.functions.pooling import max_pooling_2d
from chainer.functions.pooling import pooling_2d


class SpatialPyramidPooling2D(pooling_2d.Pooling2D):

    """Spatial pyramid pooling over a set of 2d planes."""

    def __init__(self, x_shape, pyramid_height, pooling_class, use_cudnn=True):
        bottom_c, bottom_h, bottom_w = x_shape
        self.pyramid_height = pyramid_height

        # create pooling functions for different pyramid levels
        out_dim = 0
        self.split_inds = []
        self.poolers = []
        for pyramid_level in six.moves.range(pyramid_height):
            num_bins = int(2 ** pyramid_level)

            ksize_h = int(numpy.ceil(bottom_h / (float(num_bins))))
            remainder_h = ksize_h * num_bins - bottom_h
            pad_h = remainder_h // 2

            ksize_w = int(numpy.ceil(bottom_w / (float(num_bins))))
            remainder_w = ksize_w * num_bins - bottom_w
            pad_w = remainder_w // 2

            ksize = (ksize_h, ksize_w)
            pad = (pad_h, pad_w)

            if pooling_class is max_pooling_2d.MaxPooling2D:
                pooler = pooling_class(ksize=ksize, stride=None, pad=pad,
                                       cover_all=True, use_cudnn=use_cudnn)
                self.poolers.append(pooler)
            else:
                raise NotImplementedError()

            out_dim += bottom_c * (num_bins ** 2)
            if pyramid_level < pyramid_height - 1:
                self.split_inds.append(out_dim)

    def forward(self, x):
        self.ys = []
        for pooler in self.poolers:
            y = pooler.forward(x)[0]
            n, c, h, w = pooler.out_shape = y.shape
            self.ys.append(y.reshape((n, c * h * w, 1, 1)))

        return concat.Concat(axis=1).forward(self.ys)

    def backward(self, x, gy):
        xp = cuda.get_array_module(*x)
        gx = xp.zeros_like(x[0])
        gys = xp.split(gy[0], self.split_inds, axis=1)
        for pooler, gy in zip(self.poolers, gys):
            gy = gy.reshape(pooler.out_shape)
            gx += pooler.backward(x, (gy,))[0]

        return gx,


def spatial_pyramid_pooling_2d(x, pyramid_height, pooling_class,
                               use_cudnn=True):
    """Spatial pyramid pooling function.

    It outputs a fixed-length vector regardless of input feature map size.

    It performs pooling operation to the input 4D-array ``x`` with different
    kernel sizes and padding sizes, and then flattens all dimensions except
    first dimension of all pooling results, and finally concatenates them along
    second dimension.

    At :math:`i`-th pyramid level, the kernel size
    :math:`(k_h^{(i)}, k_w^{(i)})` and padding size
    :math:`(p_h^{(i)}, p_w^{(i)})` of pooling operation are calculated as
    below:

    .. math::
        k_h^{(i)} &= \\lceil b_h / 2^i \\rceil, \\\\
        k_w^{(i)} &= \\lceil b_w / 2^i \\rceil, \\\\
        p_h^{(i)} &= (2^i k_h^{(i)} - b_h) / 2, \\\\
        p_w^{(i)} &= (2^i k_w^{(i)} - b_w) / 2,

    where :math:`\\lceil \\cdot \\rceil` denotes the ceiling function, and
    :math:`b_h, b_w` are height and width of input variable ``x``,
    respectively. Note that index of pyramid level :math:`i` is zero-based.

    See detail in paper: `Spatial Pyramid Pooling in Deep Convolutional \
    Networks for Visual Recognition \
    <https://arxiv.org/abs/1406.4729>`_.

    Args:
        x (~chainer.Variable): Input variable. The shape of ``x`` should be
            ``(batchsize, # of channels, height, width)``.
        pyramid_height (int): Number of pyramid levels
        pooling_class (MaxPooling2D or AveragePooling2D):
            Only MaxPooling2D class can be available for now.
        use_cudnn (bool): If ``True`` and cuDNN is enabled, then this function
            uses cuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable. The shape of the output variable
            will be :math:`(batchsize, c \\sum_{h=0}^{H-1} 2^{2h}, 1, 1)`,
            where :math:`c` is the number of channels of input variable ``x``
            and :math:`H` is the number of pyramid levels.

    .. note::

        This function uses some pooling classes as components to perform
        spatial pyramid pooling. Now it supports only
        :class:`~functions.MaxPooling2D` as elemental pooling operator so far.

    """

    return SpatialPyramidPooling2D(x.shape[1:], pyramid_height,
                                   pooling_class, use_cudnn=use_cudnn)(x)
